from typing import Tuple, Optional, Any
from numpy import ndarray, bool_, uint8
from dolomite_base import stage_object
import os

from .choose_dense_chunk_sizes import choose_dense_chunk_sizes
from ._utils import _translate_array_type, _open_writeable_hdf5_handle, _choose_file_dtype


@stage_object.register
def stage_ndarray(
    x: ndarray, 
    dir: str, 
    path: str, 
    is_child: bool = False, 
    chunks: Optional[Tuple[int, ...]] = None, 
    cache_size: int = 1e8, 
    **kwargs
) -> dict[str, Any]:
    """Method for saving :py:class:`~numpy.ndarray` objects to file, see
    :py:meth:`~dolomite_base.stage_object.stage_object` for details.

    Args:
        x: Array to be saved.

        dir: Staging directory.

        path: Relative path inside ``dir`` to save the object.

        is_child: Is ``x`` a child of another object?

        chunks: 
            Chunk dimensions. If not provided, we choose some chunk sizes with 
            :py:meth:`~dolomite_matrix.choose_dense_chunk_sizes.choose_dense_chunk_sizes`.

        cache_size:
            Size of the HDF5 cache size, in bytes. Larger values improve speed
            at the cost of memory.

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    os.mkdir(os.path.join(dir, path))
    newpath = path + "/array.h5"

    # Coming up with a decent chunk size.
    if chunks is None:
        chunks = choose_dense_chunk_sizes(x.shape, x.dtype.itemsize)
    else:
        capped = []
        for i, d in enumerate(x.shape):
            capped.append(min(d, chunks[i]))
        chunks = (*capped,)

    # Transposing it so that we save it in the right order.
    t = x.T
    chunks = (*list(reversed(chunks)),)

    fpath = os.path.join(dir, newpath)
    with _open_writeable_hdf5_handle(fpath, cache_size) as fhandle:
        fhandle.create_dataset("data", data=t, chunks=chunks, dtype=_choose_file_dtype(t.dtype), compression="gzip")

    return { 
        "$schema": "hdf5_dense_array/v1.json",
        "path": newpath,
        "is_child": is_child,
        "array": {
            "type": _translate_array_type(x.dtype),
            "dimensions": list(x.shape),
        },
        "hdf5_dense_array": {
            "dataset": "data",
        } 
    }
