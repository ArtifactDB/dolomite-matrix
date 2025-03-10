from typing import Dict, Any
import numpy
import os
import h5py
from hdf5array import Hdf5DenseArraySeed

from .DelayedMask import DelayedMask
from .ReloadedArray import ReloadedArray


def read_dense_array(path: str, metadata: Dict[str, Any], **kwargs) -> ReloadedArray:
    """
    Read a dense array from its on-disk representation. In general, this
    function should not be called directly but instead be dispatched via
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path: Path to the directory containing the object.

        metadata: Metadata for this object.

        kwargs: Further arguments, ignored.

    Returns:
        A :py:class:`~dolomite_matrix.ReloadedArray.ReloadedArray` containing a
        HDF5-backed dense array as a seed.
    """
    fpath = os.path.join(path, "array.h5")
    name = "dense_array"

    with h5py.File(fpath, "r") as handle:
        ghandle = handle[name]

        transposed = False
        if "transposed" in ghandle.attrs:
            transposed = (ghandle.attrs["transposed"] != 0)

        tt = ghandle.attrs["type"]
        if tt == "vls":
            return ReloadedArray(_read_vls_array(ghandle, transposed), path)

        dhandle = ghandle["data"]
        dtype = None
        if tt == "boolean":
            dtype = numpy.dtype("bool")
        elif tt == "string":
            dtype_name = "U" + str(dhandle.dtype.itemsize)
            dtype = numpy.dtype(dtype_name)
        elif tt == "float":
            if not numpy.issubdtype(dhandle.dtype, numpy.floating):
                dtype = numpy.dtype("float64")

        placeholder = None
        if "missing-value-placeholder" in dhandle.attrs:
            placeholder = dhandle.attrs["missing-value-placeholder"]

    dname = name + "/data"
    is_native = not transposed
    if placeholder is None:
        seed = Hdf5DenseArraySeed(fpath, dname, dtype=dtype, native_order=is_native)
    else: 
        core = Hdf5DenseArraySeed(fpath, dname, native_order=is_native)
        seed = DelayedMask(core, placeholder=placeholder, dtype=dtype)

    return ReloadedArray(seed, path)


# TODO: make this into a delayedarray.
def _read_vls_array(ghandle: h5py.Group, transposed: bool) -> numpy.ndarray:
    phandle = ghandle["pointers"]
    pointers = phandle[:].ravel(order="C")
    heap = ghandle["heap"][:]

    collected = []
    for i, val in enumerate(pointers):
        offset, length = val
        collected.append(bytes(heap[offset:offset + length]).decode("UTF-8"))

    payload = numpy.reshape(collected, phandle.shape, order="C")

    if "missing-value-placeholder" in phandle.attrs:
        placeholder = phandle.attrs["missing-value-placeholder"]
        if isinstance(placeholder, bytes):
            placeholder = placeholder.decode("UTF-8")
        payload = numpy.ma.array(payload, mask=(payload==placeholder))

    if transposed:
        payload = payload.T
    return payload
