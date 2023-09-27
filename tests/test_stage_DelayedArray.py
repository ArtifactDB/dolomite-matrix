import numpy
from dolomite_base import stage_object, write_metadata
from delayedarray import wrap
import delayedarray as da
import dolomite_matrix
from dolomite_matrix.stage_DelayedArray import _choose_block_shape
import os
import h5py
from tempfile import mkdtemp


def test_stage_DelayedArray_simple():
    y = wrap(numpy.random.rand(100, 200)) + 1

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "number"

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    copy = dset[:].T
    assert copy.dtype == y.dtype
    assert (copy == numpy.array(y)).all()


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


def test_stage_DelayedArray_low_block_size():
    # C-contiguous:
    y = wrap(numpy.random.rand(100, 200)) + 1

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8000)

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    copy = dset[:].T
    assert copy.dtype == y.dtype
    assert (copy == numpy.array(y)).all()

    # F-contiguous:
    y = wrap(numpy.asfortranarray(numpy.random.rand(100, 200))) + 1

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8000)

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    copy = dset[:].T
    assert copy.dtype == y.dtype
    assert (copy == numpy.array(y)).all()

def test_stage_DelayedArray_custom_chunks():
    # Chunky boi (I)
    x = numpy.random.rand(100, 200, 300)

    y = wrap(_ChunkyBoi(x, (10, 10, 10)))
    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8 * 5000)

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    copy = dset[:].T
    assert copy.dtype == x.dtype
    assert (copy == x).all()

    # Chunky boi (II)
    y = wrap(_ChunkyBoi(x, (1, 1, x.shape[2])))
    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar", block_size=8 * 5000)

    fpath = os.path.join(dir, meta["path"])
    handle = h5py.File(fpath, "r")
    dset = handle[meta["hdf5_dense_array"]["dataset"]]

    copy = dset[:].T
    assert copy.dtype == x.dtype
    assert (copy == x).all()
