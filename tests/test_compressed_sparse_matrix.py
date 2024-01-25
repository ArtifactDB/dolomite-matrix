import scipy.sparse
import dolomite_base as dl
import dolomite_matrix as dm
from tempfile import mkdtemp
import numpy
import delayedarray
import filebackedarray
import os


def test_compressed_sparse_matrix_csc():
    y = scipy.sparse.random(1000, 200, 0.1).tocsc()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, dm.ReloadedArray)
    assert isinstance(roundtrip.seed.seed, filebackedarray.Hdf5CompressedSparseMatrixSeed)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_csr():
    y = scipy.sparse.random(1000, 200, 0.1)
    y = y.tocsr()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_coo():
    y = scipy.sparse.random(1000, 200, 0.1)
    y = y.tocoo()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_SparseNdarray():
    y = delayedarray.SparseNdarray(
        (10, 5),
        [
            None, 
            (numpy.array([0, 8]), numpy.array([1, 20])), 
            None, 
            (numpy.array([2, 9]), numpy.array([0, 5000])), 
            None
        ]
    )
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert (numpy.array(roundtrip) == numpy.array(y)).all()


def test_compressed_sparse_matrix_integer():
    y = (scipy.sparse.random(1000, 200, 0.1) * 10).tocsc()
    y = y.astype(numpy.int32)
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_boolean():
    y = (scipy.sparse.random(1000, 200, 0.1) > 0).tocsc()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert (numpy.array(roundtrip) == y.toarray()).all()
