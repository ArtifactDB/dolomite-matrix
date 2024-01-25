from typing import Any
from functools import singledispatch
from dolomite_base import save_object, validate_saves
from delayedarray import SparseNdarray, chunk_shape
import os
import h5py
import numpy
import delayedarray

from . import _utils as ut
from . import _optimize_storage as optim


has_scipy = False
try:
    import scipy.sparse
    has_scipy = True
except:
    pass


def _choose_index_type(n: int) -> str:
    if n < 2**8:
        return "u1"
    elif n < 2**16:
        return "u2"
    elif n < 2**32: 
        return "u4"
    else:
        return "u8"


###############################################
###############################################


@singledispatch
def _h5_write_sparse_matrix(x: Any, handle, details, compressed_sparse_matrix_buffer_size, compressed_sparse_matrix_chunk_size):
    chunks = chunk_shape(x)
    chunks_per_col = x.shape[0] / chunks[0]
    chunks_per_row = x.shape[1] / chunks[1]

    # If we have to extract fewer chunks per column, we iterate by column to
    # create a CSC matrix. Otherwise we make a CSR matrix.
    if chunks_per_col < chunks_per_row:
        primary = 1
        secondary = 0
        handle.attrs["layout"] = "CSC"
    else:
        primary = 0
        secondary = 1
        handle.attrs["layout"] = "CSR"

    compressed_sparse_matrix_chunk_size = min(compressed_sparse_matrix_chunk_size, details.non_zero)
    dhandle = handle.create_dataset("data", shape = details.non_zero, dtype = details.type, compression = "gzip", chunks = compressed_sparse_matrix_chunk_size)
    if details.placeholder is not None:
        dhandle.create("missing-value-placeholder", data = details.placeholder, dtype = details.dtype)

    itype = _choose_index_type(x.shape[secondary])
    ihandle = handle.create_dataset("indices", shape = details.non_zero, dtype = itype, compression = "gzip", chunks = compressed_sparse_matrix_chunk_size)
    indptrs = numpy.zeros(x.shape[primary] + 1, dtype = numpy.uint64)
    counter = 0

    block_size = delayedarray.choose_block_size_for_1d_iteration(x, dimension=primary, memory=compressed_sparse_matrix_buffer_size)
    limit = x.shape[primary]
    subset = [None] * 2
    subset[secondary] = range(x.shape[secondary])

    for start in range(0, limit, block_size):
        end = min(limit, start + block_size)
        subset[primary] = range(start, end)
        block = delayedarray.extract_sparse_array(x, (*subset,))

        if block.contents is not None:
            # Sparse2darrays are always CSC, so if we need to save it as CSR,
            # we transpose it before we extract the contents.
            if primary == 0:
                block = block.T

            original = counter
            icollected = []
            dcollected = []

            for i, b in enumerate(block.contents):
                if b is not None:
                    counter += len(b[0])
                    icollected.append(b[0])
                    dcollected.append(ut.sanitize_for_writing(b[1], details.placeholder))
                indptrs[start + i + 1] = counter

            # Collecting everything in memory for a single write operation, avoid
            # potential issues with writing/reloading partial chunks. 
            ihandle[original : counter] = numpy.concatenate(icollected)
            dhandle[original : counter] = numpy.concatenate(dcollected)

    handle.create_dataset("indptr", data=indptrs, dtype="u8", compression="gzip", chunks=True)


if has_scipy:
    def _write_compressed_sparse_matrix(x: Any, handle, details, compressed_sparse_matrix_buffer_size, compressed_sparse_matrix_chunk_size, by_column): 
        if by_column:
            primary = 1
            secondary = 0
            handle.attrs["layout"] = "CSC"
        else:
            primary = 0
            secondary = 1
            handle.attrs["layout"] = "CSR"

        compressed_sparse_matrix_chunk_size = min(compressed_sparse_matrix_chunk_size, details.non_zero)
        itype = _choose_index_type(x.shape[secondary])
        handle.create_dataset("indices", data = x.indices, dtype = itype, compression = "gzip", chunks = compressed_sparse_matrix_chunk_size)
        handle.create_dataset("indptr", data = x.indptr, dtype = "u8", compression = "gzip", chunks = True)

        if not numpy.ma.is_masked(x.data):
            handle.create_dataset("data", data = x.data, dtype = details.type, compression = "gzip", chunks = compressed_sparse_matrix_chunk_size)
        elif not x.mask.any():
            handle.create_dataset("data", data = x.data.data, dtype = details.type, compression = "gzip", chunks = compressed_sparse_matrix_chunk_size)
        else:
            dhandle = handle.create_dataset("data", shape = details.non_zero, dtype = details.type, compression="gzip", chunks = compressed_sparse_matrix_chunk_size)
            if details.placeholder is not None:
                dhandle.create("missing-value-placeholder", data = details.placeholder, dtype = details.dtype)

            step = max(1, int(compressed_sparse_matrix_buffer_size / compressed_sparse_matrix_chunk_size)) * compressed_sparse_matrix_chunk_size
            for i in range(0, details.non_zero, step): 
                end = min(details.non_zero, i + step)
                block = x.data[i : end] # might be a view, so sanitization (and possible copying) is necessary.
                dhandle[i : end] = ut.sanitize_for_writing(block, details.placeholder)


    @_h5_write_sparse_matrix.register
    def _h5_write_sparse_matrix_from_csc_matrix(x: scipy.sparse.csc_matrix, handle, details, compressed_sparse_matrix_buffer_size,  compressed_sparse_matrix_chunk_size):
        _write_compressed_sparse_matrix(
            x,
            handle, 
            details, 
            compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size, 
            compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size, 
            by_column = True
        )


    @_h5_write_sparse_matrix.register
    def _h5_write_sparse_matrix_from_csr_matrix(x: scipy.sparse.csr_matrix, handle, details, compressed_sparse_matrix_buffer_size, compressed_sparse_matrix_chunk_size):
        _write_compressed_sparse_matrix(
            x,
            handle, 
            details, 
            compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size, 
            compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size, 
            by_column = False
        )


###############################################
###############################################


def _save_compressed_sparse_matrix(x: Any, path: str, compressed_sparse_matrix_chunk_size: int = 10000, compressed_sparse_matrix_buffer_size: int = 1e8, **kwargs):
    os.mkdir(path)
    if len(x.shape) != 2:
        raise ValueError("only 2-dimensional sparse arrays are currently supported")

    with h5py.File(os.path.join(path, "matrix.h5"), "w") as handle:
        ghandle = handle.create_group("compressed_sparse_matrix")

        if numpy.issubdtype(x.dtype, numpy.integer):
            tt = "integer"
            opts = optim.optimize_integer_storage(x)
        elif numpy.issubdtype(x.dtype, numpy.floating):
            tt = "number"
            opts = optim.optimize_float_storage(x)
        elif x.dtype == numpy.bool_:
            tt = "boolean"
            opts = optim.optimize_boolean_storage(x)
        else:
            raise NotImplementedError("cannot save sparse matrix of type '" + x.dtype.name + "'")

        ghandle.attrs["type"] = tt
        ghandle.create_dataset("shape", data = x.shape, dtype = "u8")
        _h5_write_sparse_matrix(
            x, 
            handle = ghandle, 
            details = opts, 
            compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size,
            compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size
        )

    with open(os.path.join(path, "OBJECT"), "w") as handle:
        handle.write('{ "type": "compressed_sparse_matrix", "compressed_sparse_matrix": { "version": "1.0" } }')


@save_object.register
@validate_saves
def save_compresssed_sparse_matrix_from_Sparse2darray(x: SparseNdarray, path: str, compressed_sparse_matrix_chunk_size: int = 10000, compressed_sparse_matrix_buffer_size: int = 1e8, **kwargs):
    """
    Method for saving a :py:class:`~delayedarray.SparseNdarray.SparseNdarray`
    to disk, see :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory to save ``x``.

        compressed_sparse_matrix_chunk_size 
            Chunk size for the data and indices. Larger values improve compression
            at the potential cost of reducing random access efficiency.

        compressed_sparse_matrix_buffer_size:
            Size of the buffer in bytes, for blockwise processing and writing
            to file. Larger values improve speed at the cost of memory.

        kwargs: Further arguments, ignored.

    Returns:
        ``x`` is saved to ``path``.
    """
    _save_compressed_sparse_matrix(
        x, 
        path, 
        compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size, 
        compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size, 
        **kwargs
    )


if has_scipy:
    @save_object.register
    @validate_saves
    def save_compresssed_sparse_matrix_from_scipy_csc_matrix(x: scipy.sparse.csc_matrix, path: str, compressed_sparse_matrix_chunk_size: int = 10000, compressed_sparse_matrix_buffer_size: int = 1e8, **kwargs):
        """
        Method for saving :py:class:`~scipy.sparse.csc_matrix` objects to disk,
        see :py:meth:`~dolomite_base.stage_object.stage_object` for details.

        Args:
            x: Matrix to be saved.

            path: Path to a directory to save ``x``.

            compressed_sparse_matrix_chunk_size 
                Chunk size for the data and indices. Larger values improve compression
                at the potential cost of reducing random access efficiency.

            compressed_sparse_matrix_cache_size:
                Size of the buffer in bytes, for blockwise processing and writing
                to file. Larger values improve speed at the cost of memory.

            kwargs: Further arguments, ignored.

        Returns:
            ``x`` is saved to ``path``.
        """
        return _save_compressed_sparse_matrix(
            x, 
            path, 
            compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size, 
            compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size, 
            **kwargs
        )


    @save_object.register
    @validate_saves
    def save_compresssed_sparse_matrix_from_scipy_csr_matrix(x: scipy.sparse.csr_matrix, path: str, compressed_sparse_matrix_chunk_size: int = 10000, compressed_sparse_matrix_buffer_size: int = 1e8, **kwargs):
        """
        Method for saving :py:class:`~scipy.sparse.csr_matrix` objects to disk,
        see :py:meth:`~dolomite_base.stage_object.stage_object` for details.

        Args:
            x: Matrix to be saved.

            path: Path to a directory to save ``x``.

            compressed_sparse_matrix_chunk_size 
                Chunk size for the data and indices. Larger values improve compression
                at the potential cost of reducing random access efficiency.

            compressed_sparse_matrix_cache_size:
                Size of the buffer in bytes, for blockwise processing and writing
                to file. Larger values improve speed at the cost of memory.

            kwargs: Further arguments, ignored.

        Returns:
            ``x`` is saved to ``path``.
        """
        return _save_compressed_sparse_matrix(
            x, 
            path, 
            compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size, 
            compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size, 
            **kwargs
        )


    @save_object.register
    @validate_saves
    def save_compresssed_sparse_matrix_from_scipy_coo_matrix(x: scipy.sparse.coo_matrix, path: str, compressed_sparse_matrix_chunk_size: int = 10000, compressed_sparse_matrix_buffer_size: int = 1e8, **kwargs):
        """
        Method for saving :py:class:`~scipy.sparse.coo_matrix` objects to disk,
        see :py:meth:`~dolomite_base.stage_object.stage_object` for details.

        Args:
            x: Matrix to be saved.

            path: Path to a directory to save ``x``.

            compressed_sparse_matrix_chunk_size 
                Chunk size for the data and indices. Larger values improve compression
                at the potential cost of reducing random access efficiency.

            compressed_sparse_matrix_cache_size:
                Size of the buffer in bytes, for blockwise processing and writing
                to file. Larger values improve speed at the cost of memory.

            kwargs: Further arguments, ignored.

        Returns:
            ``x`` is saved to ``path``.
        """
        return _save_compressed_sparse_matrix(
            x, 
            path, 
            compressed_sparse_matrix_chunk_size = compressed_sparse_matrix_chunk_size, 
            compressed_sparse_matrix_buffer_size = compressed_sparse_matrix_buffer_size, 
            **kwargs
        )
