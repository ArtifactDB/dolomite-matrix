from typing import Any
from numpy import bool_
from filebackedarray import Hdf5CompressedSparseMatrix
from dolomite_base import acquire_file


def load_hdf5_sparse_matrix(meta: dict[str, Any], project, **kwargs) -> Hdf5CompressedSparseMatrix:
    """
    Load a HDF5-backed sparse matrix. In general, this function should be
    called via :py:meth:`~dolomite_base.load_object.load_object`.

    Args:
        meta: Metadata for this HDF5 array.

        project: Value specifying the project of interest. This is most
            typically a string containing a file path to a staging directory
            but may also be an application-specific object that works with
            :py:meth:`~dolomite_base.acquire_file.acquire_file`.

        kwargs: Further arguments, ignored.

    Returns:
        A HDF5-backed sparse matrix.
    """
    fpath = acquire_file(project, meta["path"])
    name = meta["hdf5_sparse_matrix"]["group"]

    dtype = None
    if meta["array"]["type"] == "boolean":
        dtype = bool_

    return Hdf5CompressedSparseMatrix(fpath, name, shape=(*meta["array"]["dimensions"],), by_column=True, dtype=dtype)
