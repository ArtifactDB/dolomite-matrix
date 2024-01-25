import numpy
from dolomite_base import save_object, read_object
import dolomite_matrix as dm
import os
import scipy.sparse
import filebackedarray
import delayedarray
from tempfile import mkdtemp


def test_ReloadedArray_basic():
    y = numpy.random.rand(100, 200)
    dir = os.path.join(mkdtemp(), "foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)

    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert isinstance(roundtrip, dm.ReloadedArray)
    assert roundtrip.path == dir
    assert isinstance(roundtrip.seed.seed, filebackedarray.Hdf5DenseArraySeed)

    assert (delayedarray.extract_dense_array(roundtrip) == y).all()

    # Checking re-saves.
    dir2 = os.path.join(mkdtemp(), "foobar2")
    save_object(roundtrip, dir2)
    assert os.path.samefile(os.path.join(dir, "OBJECT"), os.path.join(dir2, "OBJECT"))
    assert os.path.samefile(os.path.join(dir, "array.h5"), os.path.join(dir2, "array.h5"))

    dir2 = os.path.join(mkdtemp(), "foobar2")
    save_object(roundtrip, dir2, reloaded_array_reuse_mode="symlink")
    assert os.readlink(os.path.join(dir2, "OBJECT")) == os.path.join(dir, "OBJECT")
    assert os.readlink(os.path.join(dir2, "array.h5")) == os.path.join(dir, "array.h5")

    # Copies along the tracer.
    with open(os.path.join(dir, "_TRACER"), "w") as handle:
        handle.write("YAY")
    dir2 = os.path.join(mkdtemp(), "foobar2")
    save_object(roundtrip, dir2, reloaded_array_reuse_mode="copy")
    assert os.stat(os.path.join(dir2, "OBJECT")).st_size == os.stat(os.path.join(dir, "OBJECT")).st_size
    assert os.stat(os.path.join(dir2, "array.h5")).st_size == os.stat(os.path.join(dir, "array.h5")).st_size
    assert os.stat(os.path.join(dir2, "_TRACER")).st_size == os.stat(os.path.join(dir, "_TRACER")).st_size

    # New save ignores the tracer.
    dir2 = os.path.join(mkdtemp(), "foobar2")
    save_object(roundtrip, dir2, reloaded_array_reuse_mode="none")
    assert os.path.exists(os.path.join(dir2, "OBJECT"))
    assert os.path.exists(os.path.join(dir2, "array.h5"))
    assert not os.path.exists(os.path.join(dir2, "_TRACER"))


def test_ReloadedArray_sparse():
    y = (scipy.sparse.random(1000, 200, 0.1) > 0.5).tocsc()
    dir = os.path.join(mkdtemp(),"foobar")
    save_object(y, dir)
    roundtrip = read_object(dir)

    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_
    assert isinstance(roundtrip, dm.ReloadedArray)
    assert roundtrip.path == dir
    assert isinstance(roundtrip.seed.seed, filebackedarray.Hdf5CompressedSparseMatrixSeed)

    as_sparse = delayedarray.extract_sparse_array(roundtrip) 
    assert isinstance(as_sparse, delayedarray.SparseNdarray)
    assert (numpy.array(as_sparse) == y.toarray()).all()

    # Checking re-saves.
    dir2 = os.path.join(mkdtemp(), "foobar2")
    save_object(roundtrip, dir2)
    assert os.path.samefile(os.path.join(dir, "OBJECT"), os.path.join(dir2, "OBJECT"))
    assert os.path.samefile(os.path.join(dir, "matrix.h5"), os.path.join(dir2, "matrix.h5"))

    dir2 = os.path.join(mkdtemp(), "foobar2")
    save_object(roundtrip, dir2, reloaded_array_reuse_mode="none")
    assert os.path.exists(os.path.join(dir2, "OBJECT"))
    assert os.path.exists(os.path.join(dir2, "matrix.h5"))
