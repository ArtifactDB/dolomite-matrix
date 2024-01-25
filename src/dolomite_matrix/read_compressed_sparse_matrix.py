import numpy
import os
import h5py
from delayedarray import DelayedArray
from filebackedarray import Hdf5CompressedSparseMatrixSeed

from .DelayedMask import DelayedMask
from .ReloadedArray import ReloadedArray


def read_compressed_sparse_matrix(path: str, **kwargs) -> DelayedArray:
    """
    Read a compressed sparse matrix from its on-disk representation. In
    general, this function should not be called directly but instead be
    dispatched via :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        kwargs: Further arguments, ignored.

    Returns:
        A HDF5-backed compressed sparse matrix.
    """
    fpath = os.path.join(path, "matrix.h5")
    name = "compressed_sparse_matrix"

    with h5py.File(fpath, "r") as handle:
        ghandle = handle[name]
        dhandle = ghandle["data"]

        tt = ghandle.attrs["type"]
        dtype = None
        if tt == "boolean":
            dtype = numpy.dtype("bool")
        elif tt == "float":
            if not numpy.issubdtype(dhandle.dtype, numpy.floating):
                dtype = numpy.dtype("float64")

        layout = ghandle.attrs["layout"]
        shape = (*[int(y) for y in ghandle["shape"]],)

        placeholder = None
        if "missing-value-placeholder" in dhandle.attrs:
            placeholder = dhandle.attrs["missing-value-placeholder"]

    bycol = (layout == "CSC")
    if placeholder is None:
        seed = Hdf5CompressedSparseMatrixSeed(fpath, name, shape=shape, by_column = bycol, dtype = dtype)
    else:
        core = Hdf5CompressedSparseMatrixSeed(fpath, name, shape=shape, by_column = bycol)
        seed = DelayedMask(core, placeholder=placeholder, dtype=dtype)

    return ReloadedArray(seed, path)
