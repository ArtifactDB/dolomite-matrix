import numpy
from dolomite_base import stage_object, write_metadata
import dolomite_matrix
import os
import h5py
import filebackedarray
from tempfile import mkdtemp


def test_stage_ndarray_number():
    y = numpy.random.rand(100, 200)

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "number"

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    assert dset.shape[0] == 200 # transposed, as expected.
    assert dset.shape[1] == 100
    assert dset.dtype == numpy.float64

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()


def test_stage_ndarray_integer():
    y = numpy.random.rand(150, 250) * 10
    y = y.astype(numpy.int32)

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "integer"

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    assert dset.shape[0] == 250 # transposed, as expected.
    assert dset.shape[1] == 150
    assert dset.dtype == numpy.int32

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()


def test_stage_ndarray_boolean():
    y = numpy.random.rand(99, 75) > 0

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "boolean"

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    assert dset.shape[0] == 75 # transposed, as expected.
    assert dset.shape[1] == 99 
    assert dset.dtype == numpy.uint8

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()
