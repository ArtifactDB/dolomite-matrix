from typing import Tuple, Optional, Any
import numpy
from dolomite_base import save_object, validate_saves
import delayedarray
import os
import h5py

from .choose_dense_chunk_sizes import choose_dense_chunk_sizes, _blockwise_write_to_hdf5
from . import _optimize_storage as optim


@save_object.register
@validate_saves
def save_dense_array_from_ndarray(
    x: numpy.ndarray, 
    path: str, 
    dense_array_chunk_dimensions: Optional[Tuple[int, ...]] = None, 
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
            :py:meth:`~dolomite_matrix.choose_dense_chunk_sizes.choose_dense_chunk_sizes`.

        dense_array_buffer_size:
            Size of the buffer in bytes, for blockwise processing and writing
            to file. Larger values improve speed at the cost of memory.

        kwargs: Further arguments, ignored.

    Returns:
        ``x`` is saved to ``path``.
    """
    os.mkdir(path)

    # Coming up with a decent chunk size.
    if dense_array_chunk_dimensions is None:
        dense_array_chunk_dimensions = choose_dense_chunk_sizes(x.shape, x.dtype.itemsize)
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
    
    fpath = os.path.join(path, "array.h5")
    with h5py.File(fpath, "w") as handle:
        ghandle = handle.create_group("dense_array")
        ghandle.attrs["type"] = tt

        if not blockwise:
            dhandle = ghandle.create_dataset("data", data=x, chunks=dense_array_chunk_dimensions, dtype=opts.type, compression="gzip")
        else:
            dhandle = ghandle.create_dataset("data", shape=x.shape, chunks=dense_array_chunk_dimensions, dtype=opts.type, compression="gzip")
            _blockwise_write_to_hdf5(dhandle, chunk_shape=dense_array_chunk_dimensions, x=x, placeholder=opts.placeholder, is_string=(tt == "string"), memory=dense_array_buffer_size)

        ghandle.create_dataset("transposed", data=0, dtype="i1")
        if opts.placeholder is not None:
            dhandle.attrs.create("missing-value-placeholder", data=opts.placeholder, dtype=opts.type)

    with open(os.path.join(path, "OBJECT"), "w") as handle:
        handle.write('{ "type": "dense_array", "dense_array": { "version": "1.0" } }')

