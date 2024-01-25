import numpy
import dolomite_base as dl
import dolomite_matrix as dm
import delayedarray as da
import os
import h5py
import filebackedarray
from tempfile import mkdtemp
import scipy.sparse


def test_save_delayed_array_simple():
    x = numpy.random.rand(100, 200)
    y = da.wrap(x) + 1
    assert isinstance(y, da.DelayedArray)

    dir = os.path.join(mkdtemp(), "foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, dm.ReloadedArray) 
    assert isinstance(roundtrip.seed.seed, filebackedarray.Hdf5DenseArraySeed)
    assert (numpy.array(roundtrip) == x + 1).all()


def test_save_delayed_array_booleans():
    x1 = numpy.random.rand(100, 200) > 0
    x2 = numpy.random.rand(100, 200) > 0
    y = numpy.logical_and(da.wrap(x1), da.wrap(x2))
    assert isinstance(y, da.DelayedArray)

    dir = os.path.join(mkdtemp(), "foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_
    assert (numpy.array(roundtrip) == numpy.logical_and(x1, x2)).all()


########################################################
########################################################


class _ChunkyBoi:
    def __init__(self, core, chunks):
        self._core = core
        self._chunks = chunks

    @property
    def dtype(self):
        return self._core.dtype

    @property
    def shape(self):
        return self._core.shape

@da.extract_dense_array.register
def extract_dense_array_ChunkyBoi(x: _ChunkyBoi, subsets):
    return da.extract_dense_array(x._core, subsets)

@da.chunk_shape.register
def chunk_shape_ChunkyBoi(x: _ChunkyBoi):
    return x._chunks


def test_delayed_array_custom_chunks():
    # Chunky boi (I)
    x = numpy.random.rand(100, 20, 30)
    y = da.wrap(_ChunkyBoi(x, (10, 10, 10)))
    dir = os.path.join(mkdtemp(), "foobar")
    dl.save_object(y, dir, dense_array_buffer_size=20 * 5000, dense_array_chunk_dimensions=(5, 20, 5))

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert (numpy.array(roundtrip) == x).all()

    # Chunky boi (II)
    y = da.wrap(_ChunkyBoi(x, (1, 1, x.shape[2])))
    dir = os.path.join(mkdtemp(), "foobar2")
    dl.save_object(y, dir, dense_array_buffer_size=20 * 5000, dense_array_chunk_dimensions=(5, 1, 10))

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert (numpy.array(roundtrip) == x).all()


def test_delayed_array_low_block_size_C_contiguous():
    x = numpy.random.rand(100, 200)
    y = da.wrap(x) + 1
    dir = os.path.join(mkdtemp(), "foobar")
    dl.save_object(y, dir, dense_array_buffer_size=8 * 1000)
    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert (numpy.array(roundtrip) == x + 1).all()


def test_delayed_array_low_block_size_F_contiguous():
    x = numpy.asfortranarray(numpy.random.rand(100, 200))
    y = da.wrap(x) + 1
    dir = os.path.join(mkdtemp(), "foobar")
    dl.save_object(y, dir, dense_array_buffer_size=8 * 1000)
    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert (numpy.array(roundtrip) == x + 1).all()


########################################################
########################################################


def test_delayed_array_sparse():
    x = scipy.sparse.random(1000, 200, 0.1).tocsc()
    y = da.wrap(x) * 10

    dir = os.path.join(mkdtemp(), "foobar")
    dl.save_object(y, dir)
    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, dm.ReloadedArray)
    assert isinstance(roundtrip.seed.seed, filebackedarray.Hdf5CompressedSparseMatrixSeed)
    assert (numpy.array(roundtrip) == x.toarray() * 10).all()
