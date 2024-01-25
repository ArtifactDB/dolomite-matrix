import numpy
import random
from dolomite_base import save_object, read_object
import dolomite_matrix as dm
import os
import h5py
import filebackedarray
import delayedarray
from tempfile import mkdtemp


def test_dense_array_number():
    y = numpy.random.rand(100, 200)
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert isinstance(roundtrip, dm.ReloadedArray)
    assert isinstance(roundtrip.seed.seed, filebackedarray.Hdf5DenseArraySeed)
    assert (numpy.array(roundtrip) == y).all()


def test_dense_array_integer():
    y = numpy.random.rand(150, 250) * 10
    y = y.astype(numpy.int32)
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert (numpy.array(roundtrip) == y).all()


def test_dense_array_boolean():
    y = numpy.random.rand(99, 75) > 0
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_
    assert (numpy.array(roundtrip) == y).all()


def test_dense_array_string():
    y = numpy.array(["Sumire", "Kanon", "Chisato", "Ren", "Keke"])
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.str_)
    assert (numpy.array(roundtrip) == y).all()


########################################################
########################################################


#def test_dense_array_number_mask():
#    y0 = numpy.random.rand(100, 200)
#    mask = y0 > 0.9
#    y = numpy.ma.MaskedArray(y0, mask=mask)
#
#    dir = os.path.join(mkdtemp(), "foobar")
#    save_object(y, dir)
#    roundtrip = read_object(dir)
#    assert roundtrip.shape == y.shape
#    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
#
#    dump = delayedarray.extract_dense_array(roundtrip)
#    assert (dump.mask == mask).all()
#    assert (dump == y).all()


########################################################
########################################################


def test_dense_array_F_contiguous():
    x = numpy.asfortranarray(numpy.random.rand(100, 200))
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(x, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == x.shape
    assert roundtrip.dtype == x.dtype
    assert (numpy.array(roundtrip) == x).all()


def test_dense_array_block_size():
    x = numpy.ndarray([100, 200], dtype="U1")
    choices = "ABCDE"
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = random.choice(choices)

    dir = os.path.join(mkdtemp(), "foobar")
    save_object(x, dir, dense_array_buffer_size= x.dtype.itemsize * 1000)
    roundtrip = read_object(dir)
    assert roundtrip.shape == x.shape
    assert roundtrip.dtype == x.dtype
    assert (numpy.array(roundtrip) == x).all()
