from typing import Any
import numpy
import os
import h5py

from filebackedarray import Hdf5DenseArray


def read_dense_array(path: str, **kwargs) -> Hdf5DenseArray:
    """
    Read a dense array from its on-disk representation. In general, this
    function should not be called directly but instead be dispatched via
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        kwargs: Further arguments, ignored.

    Returns:
        A HDF5-backed dense array.
    """
    fpath = os.path.join(path, "array.h5")
    with h5py.File(fpath, "r") as handle:
        ghandle = handle["dense_array"]
        dhandle = ghandle["data"]

        tt = ghandle.attrs["type"]
        dtype = None
        if tt == "boolean":
            dtype = numpy.dtype("bool")
        elif tt == "string":
            dtype_name = "U" + str(dhandle.dtype.itemsize)
            dtype = numpy.dtype(dtype_name)
        elif tt == "float":
            if not numpy.issubdtype(dhandle.dtype, numpy.floating):
                dtype = numpy.dtype("float64")

        transposed = False
        if "transposed" in ghandle:
            transposed = (ghandle["transposed"][()] != 0)

        placeholder = None
        if "missing-value-placeholder" in dhandle.attrs:
            raise NotImplementedError("oops, no support for arrays with missing values at the moment")

    return Hdf5DenseArray(fpath, "dense_array/data", dtype=dtype, native_order=not transposed)
