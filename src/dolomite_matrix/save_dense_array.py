from typing import Tuple, Optional, Any
from numpy import ndarray, bool_, uint8
from dolomite_base import save_object, validate_saves
import os

from .choose_dense_chunk_sizes import choose_dense_chunk_sizes
from ._utils import _translate_array_type, _open_writeable_hdf5_handle, _choose_file_dtype


@save_object.register
def save_dense_array_from_ndarray(
    x: ndarray, 
    path: str, 
    dense_array_chunk_dimensions: Optional[Tuple[int, ...]] = None, 
    dense_array_cache_size: int = 1e8, 
    **kwargs
):
    """Method for saving :py:class:`~numpy.ndarray` objects to their
    corresponding file representations, see
    :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory to save ``x``.

        dense_array_chunk_dimensions: 
            Chunk dimensions. If not provided, we choose some chunk sizes with
            :py:meth:`~dolomite_matrix.choose_dense_chunk_sizes.choose_dense_chunk_sizes`.

        dense_array_cache_size:
            Size of the HDF5 cache size, in bytes. Larger values improve speed
            at the cost of memory.

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    os.mkdir(path)
    newpath = path + "/array.h5"

    # Coming up with a decent chunk size.
    if chunks is None:
        chunks = choose_dense_chunk_sizes(x.shape, x.dtype.itemsize)
    else:
        capped = []
        for i, d in enumerate(x.shape):
            capped.append(min(d, chunks[i]))
        chunks = (*capped,)

    # Choosing the smallest data type that we can use        

    # Transposing it so that we save it in the right order.
    t = x.T
    chunks = (*list(reversed(chunks)),)

    fpath = os.path.join(dir, newpath)
    with _open_writeable_hdf5_handle(fpath, cache_size) as fhandle:
        fhandle.create_dataset("data", data=t, chunks=chunks, dtype=_choose_file_dtype(t.dtype), compression="gzip")
