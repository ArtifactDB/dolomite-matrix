from typing import Any
from numpy import bool_
from filebackedarray import Hdf5DenseArray
from dolomite_base import acquire_file


def load_hdf5_dense_array(meta: dict[str, Any], project, **kwargs) -> Hdf5DenseArray:
    """
    Load a HDF5-backed dense array.  In general, this function should be
    called via :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this HDF5 array.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, ignored.

    Returns:
        A HDF5-backed dense array.
    """
    fpath = acquire_file(project, meta["path"])
    name = meta["hdf5_dense_array"]["dataset"]

    dtype = None
    if meta["array"]["type"] == "boolean":
        dtype = bool_

    return Hdf5DenseArray(fpath, name, dtype=dtype, native_order=False)
