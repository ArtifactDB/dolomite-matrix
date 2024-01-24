import scipy.sparse
import dolomite_base as dl
import dolomite_matrix as dm
from tempfile import mkdtemp
import numpy
import filebackedarray
import os


def test_compressed_sparse_matrix_csc():
    y = scipy.sparse.random(1000, 200, 0.1).tocsc()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dm.read_compressed_sparse_matrix(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_csr():
    y = scipy.sparse.random(1000, 200, 0.1)
    y = y.tocsr()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dm.read_compressed_sparse_matrix(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_coo():
    y = scipy.sparse.random(1000, 200, 0.1)
    y = y.tocoo()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dm.read_compressed_sparse_matrix(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_integer():
    y = (scipy.sparse.random(1000, 200, 0.1) * 10).tocsc()
    y = y.astype(numpy.int32)
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dm.read_compressed_sparse_matrix(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_compressed_sparse_matrix_boolean():
    y = (scipy.sparse.random(1000, 200, 0.1) > 0).tocsc()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dm.read_compressed_sparse_matrix(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()
