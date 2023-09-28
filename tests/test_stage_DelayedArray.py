import numpy
from dolomite_base import stage_object, write_metadata
from delayedarray import wrap
import delayedarray as da
import dolomite_matrix
from dolomite_matrix._utils import _choose_block_shape
import os
import h5py
import filebackedarray
from tempfile import mkdtemp


def test_stage_DelayedArray_simple():
    x = numpy.random.rand(100, 200)
    y = wrap(x) + 1
    assert isinstance(y, da.DelayedArray)

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "number"

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    copy = dset[:].T
    assert copy.dtype == y.dtype
    assert (copy == x + 1).all()

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == x + 1).all()


def test_stage_DelayedArray_booleans():
    x1 = numpy.random.rand(100, 200) > 0
    x2 = numpy.random.rand(100, 200) > 0
    y = numpy.logical_and(wrap(x1), wrap(x2))
    assert isinstance(y, da.DelayedArray)

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "boolean"

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
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


########################################################
########################################################


def test_stage_DelayedArray_choose_block_shape():
    y = _ChunkyBoi(numpy.random.rand(100, 200), (10, 10))
    assert _choose_block_shape(y, 2000 * 8) == (10, 200)
    assert _choose_block_shape(y, 5000 * 8) == (20, 200)

    y = _ChunkyBoi(numpy.random.rand(100, 200), (100, 10))
    assert _choose_block_shape(y, 2000 * 8) == (100, 20)
    assert _choose_block_shape(y, 5000 * 8) == (100, 50)

    y = _ChunkyBoi(numpy.random.rand(100, 200, 300), (10, 10, 10))
    assert _choose_block_shape(y, 2000 * 8) == (10, 10, 20)

    y = _ChunkyBoi(numpy.random.rand(100, 200, 300), (1, 1, 300))
    assert _choose_block_shape(y, 5000 * 8) == (1, 16, 300)


def test_stage_DelayedArray_low_block_size_C_contiguous():
    x = numpy.random.rand(100, 200)
    y = wrap(x) + 1

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8000)

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == x + 1).all()


def test_stage_DelayedArray_low_block_size_F_contiguous():
    x = numpy.asfortranarray(numpy.random.rand(100, 200))
    y = wrap(x) + 1

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8000)

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == x + 1).all()


def test_stage_DelayedArray_custom_chunks():
    # Chunky boi (I)
    x = numpy.random.rand(100, 200, 300)

    y = wrap(_ChunkyBoi(x, (10, 10, 10)))
    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8 * 5000)

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == x).all()

    # Chunky boi (II)
    y = wrap(_ChunkyBoi(x, (1, 1, x.shape[2])))
    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8 * 5000)

    roundtrip = dolomite_matrix.load_hdf5_dense_array(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5DenseArray)
    assert (numpy.array(roundtrip) == x).all()
