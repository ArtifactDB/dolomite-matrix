import scipy.sparse
from dolomite_base import stage_object, write_metadata
import dolomite_matrix
from tempfile import mkdtemp
import filebackedarray
import numpy


def test_stage_scipy_csc():
    y = scipy.sparse.random(1000, 200, 0.1).tocsc()

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "number"

    roundtrip = dolomite_matrix.load_hdf5_sparse_matrix(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_stage_scipy_csr():
    y = scipy.sparse.random(1000, 200, 0.1) * 10
    y = y.astype(numpy.int32)
    y = y.tocsr()

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "integer"

    roundtrip = dolomite_matrix.load_hdf5_sparse_matrix(meta, dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()


def test_stage_ndarray_boolean():
    y = scipy.sparse.random(1000, 200, 0.1) > 0
    y = y.tocoo()

    dir = mkdtemp()
    meta = stage_object(y, dir=dir, path="foobar")
    write_metadata(meta, dir=dir)
    assert meta["array"]["type"] == "boolean"

    roundtrip = dolomite_matrix.load_hdf5_sparse_matrix(meta, dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, filebackedarray.Hdf5CompressedSparseMatrix)
    assert (numpy.array(roundtrip) == y.toarray()).all()
