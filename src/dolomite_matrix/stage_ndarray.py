from typing import Tuple, Optional
from numpy import ndarray, bool_
from dolomite_base import stage_object
from h5py import File

from .guess_dense_chunk_sizes import guess_dense_chunk_sizes
from _utils import _translate_array_type


@stage_object.register
def stage_ndarray(x: ndarray, dir: str, path: str, is_child: bool = False, chunks: Optional[Tuple[int]] = None, **kwargs) -> dict[str, Any]:
    """Method for saving :py:class:`~numpy.ndarray` objects to file, see
    :py:meth:`~dolomite_base.stage_object.stage_object` for details.

    Args:
        x: Array to be saved.

        dir: Staging directory.

        path: Relative path inside ``dir`` to save the object.

        is_child: Is ``x`` a child of another object?

        chunks: 
            Chunk dimensions. If not provided, we guess some chunk sizes with 
            `:py:meth:`~dolomite_matrix.guess_dense_chunk_sizes.guess_dense_chunk_sizes`.

        kwargs: Further arguments, ignored.

    Returns:
        Metadata that can be edited by calling methods and then saved with 
        :py:meth:`~dolomite_base.write_metadata.write_metadata`.
    """
    os.mkdir(os.path.join(dir, path))
    newpath = path + "/array.h5"

    fpath = os.path.join(dir, newpath)
    fhandle = File(fpath, "w", )

    # Transposing it so that we save it in the right order.
    t = x.T

    # Coming up with a decent chunk size.
    chunks = guess_dense_chunk_sizes(t.shape, x.dtype.itemsize)

    savetype = t.dtype
    if savetype == numpy.bool_:
        savetype = numpy.uint8

    f.create_dataset("data", data=t, chunks=chunks, dtype=savetype)

    return { 
        "$schema": "hdf5_dense_array/v1.json",
        "path": newpath,
        "is_child": is_child,
        "array" = {
            "type": _translate_array_type(x),
            "dimensions": x.shape,
        },
        "hdf5_dense_array" = {
            "dataset": "data",
        )
    }
