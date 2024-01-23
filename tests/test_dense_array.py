import numpy
from dolomite_base import save_object
import dolomite_matrix as dm
import os
import h5py
import filebackedarray
from tempfile import mkdtemp


def test_dense_array_number():
    y = numpy.random.rand(100, 200)
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = dm.read_dense_array(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()


def test_dense_array_integer():
    y = numpy.random.rand(150, 250) * 10
    y = y.astype(numpy.int32)
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = dm.read_dense_array(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()


def test_dense_array_boolean():
    y = numpy.random.rand(99, 75) > 0
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = dm.read_dense_array(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()


def test_dense_array_string():
    y = numpy.array(["Sumire", "Kanon", "Chisato", "Ren", "Keke"])
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = dm.read_dense_array(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.str_)
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == y).all()

