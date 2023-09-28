from functools import singledispatch
from typing import Any
from collections import namedtuple
from delayedarray import wrap, extract_sparse_array, DelayedArray, SparseNdarray, Combine, guess_iteration_block_size
from h5py import File
import numpy

from ._utils import _choose_block_shape


def _choose_smallest_uint(value):
    if value < 2**8:
        return numpy.uint8
    elif value < 2**16:
        return numpy.uint16
    elif value < 2**32:
        return numpy.uint32
    else:
        return numpy.uint64


def write_sparse_matrix(x, path: str, name: str, chunks: int = 10000, guess_integer: bool = True, block_size: int = 1e8):
    """Write a sparse matrix into a HDF5 file in a compressed-sparse column
    format. This creates a HDF5 group containing the ``data``, ``indices`` and
    ``indptr`` datasets, containing the corresponding components of the usual
    compressed sparse representation. The shape of the matrix is stored in the
    ``shape`` dataset inside the same group.

    Args:
        x: Any sparse matrix.

        path: Path to the file in which to save ``x``.

        name: Name of the group in the file.

        chunks: Size of the chunks for the data and index datasets.

        guess_integer: 
            Whether to guess a compact integer type for the data. This can reduce
            disk usage at the cost of a second pass through the data.

        block_size:
            Size of the blocks to use for processing ``x``. Larger values
            increase speed at the cost of memory efficiency.

    Returns:
        A HDF5 file is created at ``path`` with the sparse data stored in the
        ``name`` group.
    """
    with File(path, "w") as handle:
        ghandle = handle.create_group(name)
        ghandle.create_dataset(name="shape", data=numpy.array(x.shape, dtype=numpy.uint64))

        # Choosing a compact type for the indices.
        index_dtype = _choose_smallest_uint(x.shape[0])

        deets = _extract_details(x, block_size=block_size)

        # Choosing a compact type for the data.
        if not guess_integer:
            dtype = _choose_file_dtype(x.dtype)
        else:
            if deets.non_integer:
                if x.dtype == numpy.float32:
                    dtype = numpy.float32
                else:
                    dtype = numpy.float64
            elif deets.minimum < 0:
                if deets.minimum >= -2**7 and deets.maximum < 2**7:
                    dtype = numpy.int8
                elif deets.minimum >= -2**15 and deets.maximum < 2**15:
                    dtype = numpy.int16
                elif deets.minimum >= -2**31 and deets.maximum < 2**31:
                    dtype = numpy.int32
                else:
                    dtype = numpy.int64
            else:
                dtype = _choose_smallest_uint(deets.maximum)

        # Doing the dump.
        ihandle = ghandle.create_dataset(name="indices", shape=deets.count, dtype=index_dtype, chunks=chunks, compression="gzip")
        dhandle = ghandle.create_dataset(name="data", shape=deets.count, dtype=dtype, chunks=chunks, compression="gzip")
        saved = _dump_sparse_matrix(x, ihandle, dhandle, block_size=block_size, offset=0)

        pointers = numpy.zeros(x.shape[1] + 1, dtype=numpy.uint64)
        for i, s in enumerate(saved):
            pointers[i + 1] += pointers[i] + s
        ghandle.create_dataset(name="indptr", data=pointers, compression="gzip")


has_scipy = False
try:
    import scipy.sparse
    has_scipy = True
except:
    pass


####################################################
####################################################


_SparseMatrixDetails = namedtuple("_SparseMatrixDetails", [ "minimum", "maximum", "non_integer", "count" ])


def _extract_details_by_block(x, block_size: int) -> _SparseMatrixDetails:
    full_shape = x.shape
    block_shape = _choose_block_shape(x, block_size)

    num_row_blocks = int(numpy.ceil(full_shape[0] / block_shape[0]))
    num_col_blocks = int(numpy.ceil(full_shape[1] / block_shape[1]))

    minimum = None
    maximum = None
    count = 0
    non_integer = False

    for c in range(num_col_blocks):
        col_start = c * block_shape[1]
        col_range = range(col_start, min(col_start + block_shape[1], full_shape[1]))

        for r in range(num_row_blocks):
            row_start = c * block_shape[0]
            row_range = range(row_start, min(row_start + block_shape[0], full_shape[0]))

            block = extract_sparse_array(x, (row_range, col_range))
            deets = _extract_details(block)

            if minimum is None or deets.minimum < minimum:
                minimum = deets.minimum
            if maximum is None or deets.maximum > maximum:
                maximum = deets.maximum
            if not non_integer:
                non_integer = deets.non_integer
            count += deets.count

    return _SparseMatrixDetails(
        minimum = minimum, 
        maximum = maximum, 
        non_integer = non_integer, 
        count = count,
    )


@singledispatch
def _extract_details(x: Any, block_size: int, **kwargs) -> _SparseMatrixDetails:
    return _extract_details_by_block(x, block_size)


def _has_non_integer(data: numpy.ndarray):
    if numpy.issubdtype(data.dtype, numpy.floating):
        for y in data:
            if not y.is_integer():
                return True
    return False


@_extract_details.register
def _extract_details_SparseNdarray(x: SparseNdarray, **kwargs) -> _SparseMatrixDetails:
    minimum = None
    maximum = None
    count = 0
    non_integer = False

    if x.contents is not None:
        for idx, val in x.contents:
            candidate_min = min(val)
            if minimum is None or candidate_min < minimum:
                minimum = candidate_min

            candidate_max = max(val)
            if maximum is None or candidate_max > maximum:
                maximum = candidate_max

            if not non_integer:
                non_integer = _has_non_integer(val)

            count += len(val)
    else:
        minimum = 0
        maximum = 0

    return _SparseMatrixDetails(
        minimum = minimum, 
        maximum = maximum, 
        non_integer = non_integer, 
        count = count,
    )



@_extract_details.register
def _extract_details_Combine(x: Combine, **kwargs) -> _SparseMatrixDetails:
    minimum = None
    maximum = None
    count = 0
    non_integer = False

    for s in x.seeds:
        deets = _extract_details(s, **kwargs)
        if minimum is None or deets.minimum < minimum:
            minimum = deets.minimum
        if maximum is None or deets.maximum > maximum:
            maximum = deets.maximum
        if not non_integer:
            non_integer = deets.non_integer
        count += deets.count

    return _SparseMatrixDetails(
        minimum = minimum, 
        maximum = maximum, 
        non_integer = non_integer, 
        count = count,
    )


@_extract_details.register
def _extract_details_DelayedArray(x: DelayedArray, block_size: int, **kwargs) -> _SparseMatrixDetails:
    candidate = _extract_details.dispatch(type(x.seed))
    if _extract_details.dispatch(object) != candidate:
        return candidate(x.seed, block_size = block_size, **kwargs)
    else:
        return _extract_details_by_block(x, block_size)


if has_scipy:
    def _extract_from_compressed_data(data: numpy.ndarray):
        return _SparseMatrixDetails(
            minimum = min(data),
            maximum = max(data),
            non_integer = _has_non_integer(data),
            count = len(data),
        )


    @_extract_details.register
    def _extract_details_scipy_csc(x: scipy.sparse.csc_matrix, **kwargs):
        return _extract_from_compressed_data(x.data)


    @_extract_details.register
    def _extract_details_scipy_csr(x: scipy.sparse.csr_matrix, **kwargs):
        return _extract_from_compressed_data(x.data)


    @_extract_details.register
    def _extract_details_scipy_coo(x: scipy.sparse.coo_matrix, **kwargs):
        return _extract_from_compressed_data(x.data)


####################################################
####################################################

def _dump_sparse_matrix_as_SparseNdarray(x, indices_handle, data_handle, offset: int): 
    saved = numpy.zeros(x.shape[1], dtype=numpy.uint64)
    if x.contents is not None:
        for s, con in enumerate(x.contents):
            if con is not None:
                idx, val = con
                indices_handle[offset:offset + len(idx)] = idx
                data_handle[offset:offset + len(val)] = val
                offset += len(idx)
                saved[s] = len(idx)
    return saved


def _dump_sparse_matrix_by_block(x, indices_handle, data_handle, offset: int, block_size: int):
    block_size = guess_iteration_block_size(x, 1, block_size)
    start = 0 
    row_range = range(x.shape[0])
    all_saved = [numpy.zeros(0, dtype=numpy.uint64)]

    while start < x.shape[1]:
        end = min(start + block_size, x.shape[1])
        col_range = range(start, end)
        block = extract_sparse_array(x, (row_range, col_range))
        saved = _dump_sparse_matrix_as_SparseNdarray(block, indices_handle, data_handle, offset=offset)
        offset += saved.sum()
        all_saved.append(saved)
        start = end

    return numpy.concatenate(all_saved)


@singledispatch
def _dump_sparse_matrix(x: Any, indices_handle, data_handle, offset: int, block_size: int, **kwargs) -> numpy.ndarray:
    return _dump_sparse_matrix_by_block(x, indices_handle, data_handle, offset=offset, block_size=block_size)


@_dump_sparse_matrix.register
def _dump_sparse_matrix_SparseNdarray(x: SparseNdarray, indices_handle, data_handle, offset: int, block_size: int, **kwargs):
    return _dump_sparse_matrix_by_block(x, indices_handle, data_handle, offset=offset, block_size=block_size)


@_dump_sparse_matrix.register
def _dump_sparse_matrix_Combine(x: Combine, indices_handle, data_handle, offset: int, block_size: int, **kwargs):
    if x.along != 1:
        return _dump_sparse_matrix_by_block(x, indices_handle, data_handle, offset=offset, block_size=block_size)

    all_saved = [numpy.zeros(0, dtype=numpy.uint64)]
    offset = 0
    for s in x.seeds:
        saved = _dump_sparse_matrix(s, indices_handle, data_handle, offset=offset, block_size=block_size, **kwargs)
        all_saved.append(saved)
        offset += int(saved.sum())

    return numpy.concatenate(all_saved)


@_dump_sparse_matrix.register
def _dump_sparse_matrix_DelayedArray(x: DelayedArray, indices_handle, data_handle, offset: int, block_size: int, **kwargs):
    candidate = _dump_sparse_matrix.dispatch(type(x.seed))
    if _dump_sparse_matrix.dispatch(object) != candidate:
        return candidate(x.seed, indices_handle, data_handle, offset=offset, block_size = block_size, **kwargs)
    else:
        return _dump_sparse_matrix_by_block(x, indices_handle, data_handle, offset=offset, block_size=block_size)


if has_scipy:
    # All other representations can undergo block processing.
    @_dump_sparse_matrix.register
    def _dump_sparse_matrix_scipy_csc(x: scipy.sparse.csc_matrix, indices_handle, data_handle, offset: int, block_size: int, **kwargs):
        n = len(x.data)
        indices_handle[offset:offset + n] = x.indices
        data_handle[offset:offset + n] = x.data
        saved = numpy.zeros(x.shape[1], dtype=numpy.uint64)
        for i in range(x.shape[1]):
            saved[i] = x.indptr[i+1] - x.indptr[i]
        return saved
