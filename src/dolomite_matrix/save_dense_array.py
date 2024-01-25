from typing import Tuple, Optional, Any, Dict
import numpy
from dolomite_base import save_object, validate_saves
import delayedarray
import os
import h5py

from .choose_chunk_dimensions import choose_chunk_dimensions 
from . import _optimize_storage as optim
from . import _utils as ut


###################################################
###################################################


# We use a mock class with the properties of the HDF5 dataset.  This allows us
# to use choose_block_shape_for_iteration to pick dimensions that align with
# the HDF5 chunks; these may or may not be suitable for the input array, but 
# we'll take the chance that the input array is already in memory.
class _DenseArrayOutputMock:
    def __init__(self, shape: Tuple, dtype: numpy.dtype, chunks: Tuple):
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks


@delayedarray.chunk_shape.register
def _chunk_shape_DenseArrayOutputMock(x: _DenseArrayOutputMock):
    return x.chunks


def _blockwise_write_to_hdf5(dhandle: h5py.Dataset, chunk_shape: Tuple, x: Any, placeholder: Any, is_string: bool, memory: int):
    mock = _DenseArrayOutputMock(x.shape, x.dtype, chunk_shape)
    block_shape = delayedarray.choose_block_shape_for_iteration(mock, memory=memory)
    if placeholder is not None:
        placeholder = x.dtype.type(placeholder)

    def _blockwise_dense_writer(pos: Tuple, block):
        block = ut.sanitize_for_writing(block, placeholder)

        # h5py doesn't want to convert from numpy's Unicode type to bytes
        # automatically, and fails: so fine, we'll do it ourselves.
        if is_string: 
            block = block.astype(dhandle.dtype, copy=False)

        # Block processing is inherently Fortran-order based (i.e., first
        # dimension is assumed to change the fastest), and the blocks
        # themselves are also in F-contiguous layout (i.e., column-major). By
        # comparison HDF5 uses C order. To avoid any rearrangement of data
        # by h5py, we save it as a transposed array for efficiency.
        coords = [slice(start, end) for start, end in reversed(pos)]
        dhandle[(*coords,)] = block.T

    delayedarray.apply_over_blocks(x, _blockwise_dense_writer, block_shape = block_shape)
    return


###################################################
###################################################


def _save_dense_array(
    x: numpy.ndarray, 
    path: str, 
    dense_array_chunk_dimensions: Optional[Tuple[int, ...]] = None, 
    dense_array_chunk_args: Dict = {},
    dense_array_buffer_size: int = 1e8, 
    **kwargs
):
    os.mkdir(path)

    # Coming up with a decent chunk size.
    if dense_array_chunk_dimensions is None:
        dense_array_chunk_dimensions = choose_chunk_dimensions(x.shape, x.dtype.itemsize, **dense_array_chunk_args)
    else:
        capped = []
        for i, d in enumerate(x.shape):
            capped.append(min(d, dense_array_chunk_dimensions[i]))
        dense_array_chunk_dimensions = (*capped,)

    # Choosing the smallest data type that we can use.
    tt = None
    blockwise = False 
    if numpy.issubdtype(x.dtype, numpy.integer):
        tt = "integer"
        opts = optim.optimize_integer_storage(x)
    elif numpy.issubdtype(x.dtype, numpy.floating):
        tt = "number"
        opts = optim.optimize_float_storage(x)
    elif x.dtype == numpy.bool_:
        tt = "boolean"
        opts = optim.optimize_boolean_storage(x)
    elif numpy.issubdtype(x.dtype, numpy.str_):
        tt = "string"
        opts = optim.optimize_string_storage(x)
        blockwise = True
    else:
        raise NotImplementedError("cannot save dense array of type '" + x.dtype.name + "'")

    if opts.placeholder is not None:
        blockwise = True
    if not isinstance(x, numpy.ndarray):
        blockwise = True
    
    fpath = os.path.join(path, "array.h5")
    with h5py.File(fpath, "w") as handle:
        ghandle = handle.create_group("dense_array")
        ghandle.attrs["type"] = tt

        if not blockwise:
            # Saving it in transposed form if it's in Fortran order (i.e., first dimensions are fastest).
            # This avoids the need for any data reorganization inside h5py itself.
            if x.flags.f_contiguous:
                x = x.T
                dense_array_chunk_dimensions = (*reversed(dense_array_chunk_dimensions),)
                ghandle.create_dataset("transposed", data=1, dtype="i1")
            else:
                ghandle.create_dataset("transposed", data=0, dtype="i1")
            dhandle = ghandle.create_dataset("data", data=x, chunks=dense_array_chunk_dimensions, dtype=opts.type, compression="gzip")
        else:
            # Block processing of a dataset is always Fortran order, but HDF5 uses C order.
            # So, we save the blocks in transposed form for efficiency.
            ghandle.create_dataset("transposed", data=1, dtype="i1")
            dhandle = ghandle.create_dataset("data", shape=(*reversed(x.shape),), chunks=(*reversed(dense_array_chunk_dimensions),), dtype=opts.type, compression="gzip")
            _blockwise_write_to_hdf5(dhandle, chunk_shape=dense_array_chunk_dimensions, x=x, placeholder=opts.placeholder, is_string=(tt == "string"), memory=dense_array_buffer_size)
            if opts.placeholder is not None:
                dhandle.attrs.create("missing-value-placeholder", data=opts.placeholder, dtype=opts.type)

    with open(os.path.join(path, "OBJECT"), "w") as handle:
        handle.write('{ "type": "dense_array", "dense_array": { "version": "1.0" } }')


###################################################
###################################################


@save_object.register
@validate_saves
def save_dense_array_from_ndarray(
    x: numpy.ndarray, 
    path: str, 
    dense_array_chunk_dimensions: Optional[Tuple[int, ...]] = None, 
    dense_array_chunk_args: Dict = {},
    dense_array_buffer_size: int = 1e8, 
    **kwargs
):
    """
    Method for saving :py:class:`~numpy.ndarray` objects to disk, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory to save ``x``.

        dense_array_chunk_dimensions: 
            Chunk dimensions for the HDF5 dataset. Larger values improve
            compression at the potential cost of reducing random access
            efficiency. If not provided, we choose some chunk sizes with
            :py:meth:`~dolomite_matrix.choose_chunk_dimensions.choose_chunk_dimensions`.

        dense_array_chunk_args: 
            Arguments to pass to ``choose_chunk_dimensions`` if
            ``dense_array_chunk_dimensions`` is not provided.

        dense_array_buffer_size:
            Size of the buffer in bytes, for blockwise processing and writing
            to file. Larger values improve speed at the cost of memory.

        kwargs: Further arguments, ignored.

    Returns:
        ``x`` is saved to ``path``.
    """
    _save_dense_array(
        x, 
        path=path, 
        dense_array_chunk_dimensions=dense_array_chunk_dimensions,
        dense_array_chunk_args = dense_array_chunk_args,
        dense_array_buffer_size = dense_array_buffer_size,
        **kwargs
    )
