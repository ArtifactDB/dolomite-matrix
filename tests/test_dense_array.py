import numpy
import random
from dolomite_base import save_object, read_object
import dolomite_matrix as dm
import os
import h5py
import hdf5array
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
    assert isinstance(roundtrip.seed.seed, hdf5array.Hdf5DenseArraySeed)
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


def test_dense_array_number_mask():
    y0 = numpy.random.rand(100, 200)
    mask = y0 > 0.9
    y = numpy.ma.MaskedArray(y0, mask=mask)

    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)

    dump = delayedarray.to_dense_array(roundtrip)
    assert (dump.mask == mask).all()
    assert numpy.logical_or(dump == y, mask).all()


def test_dense_array_number_mask_complex():
    y0 = numpy.random.rand(100, 200)
    mask = y0 > 0.9
    y = numpy.ma.MaskedArray(y0, mask=mask)
    y[0, 0] = numpy.nan
    y[1, 1] = numpy.inf
    y[2, 2] = -numpy.inf

    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)

    dump = delayedarray.to_dense_array(roundtrip)
    assert (dump.mask == mask).all()
    vals = numpy.logical_or(dump == y, numpy.isnan(dump) == numpy.isnan(y))
    assert numpy.logical_or(vals, mask).all()


def test_dense_array_number_mask_integer():
    # This setup is chosen to populate all of the 8-bit space in the non-masked
    # data, which subsequently forces a type promotion during blockwise writing
    # so that we can correctly insert the placeholder.
    y0 = (numpy.random.rand(100, 200) * 256).astype(numpy.uint8)
    mask = numpy.random.rand(100, 200) < 0.5
    y = numpy.ma.MaskedArray(y0, mask=mask)

    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)

    dump = delayedarray.to_dense_array(roundtrip)
    assert (dump.mask == mask).all()
    assert numpy.logical_or(dump == y, mask).all()


def test_dense_array_number_mask_string():
    # This setup is chosen to force promotion to a longer string length during
    # blockwise writing so that we correctly insert the placeholder.
    x = numpy.ndarray([100, 200], dtype="U1")
    choices = "abcdefghijk"
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = random.choice(choices)

    mask = numpy.random.rand(100, 200) < 0.5
    x = numpy.ma.MaskedArray(x, mask=mask)

    dir = os.path.join(mkdtemp(), "foobar")
    save_object(x, dir)
    roundtrip = read_object(dir)
    assert roundtrip.shape == x.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.str_)

    dump = delayedarray.to_dense_array(roundtrip)
    assert (dump.mask == mask).all()
    assert numpy.logical_or(dump == x, mask).all()


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
    # Triggering blockwise processing by using strings.
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

    # Triggering blockwise processing by using placeholders.
    x0 = numpy.random.rand(100, 200)
    mask = x0 > 0.9
    x = numpy.ma.MaskedArray(x0, mask=mask)

    dir = os.path.join(mkdtemp(), "foobar")
    save_object(x, dir, dense_array_buffer_size=x.dtype.itemsize * 50)
    roundtrip = read_object(dir)
    assert roundtrip.shape == x.shape
    assert roundtrip.dtype == x.dtype
    assert (numpy.array(roundtrip) == x).all()
