import scipy.sparse
import dolomite_matrix
from tempfile import mkstemp
from filebackedarray import Hdf5CompressedSparseMatrix
from delayedarray import wrap, Combine
import delayedarray
import h5py
import numpy


def test_write_sparse_matrix_scipy_csc():
    out = scipy.sparse.random(1000, 200, 0.1).tocsc()
    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")

    with h5py.File(fpath, "r") as handle:
        dset = handle["foo/data"]
        assert dset.dtype == numpy.float64
        iset = handle["foo/indices"]
        assert iset.dtype == numpy.uint16
        assert list(handle["foo/shape"][:]) == [1000, 200]

    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == out.toarray()).all()


def test_write_sparse_matrix_scipy_other():
    out = scipy.sparse.random(1000, 200, 0.1).tocoo()
    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")

    with h5py.File(fpath, "r") as handle:
        dset = handle["foo/data"]
        assert dset.dtype == numpy.float64
        iset = handle["foo/indices"]
        assert iset.dtype == numpy.uint16
        assert list(handle["foo/shape"][:]) == [1000, 200]

    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == out.toarray()).all()


def test_write_sparse_matrix_DelayedArray():
    raw = scipy.sparse.random(1000, 200, 0.1).tocsc()
    out = wrap(raw) * 10
    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")

    with h5py.File(fpath, "r") as handle:
        dset = handle["foo/data"]
        assert dset.dtype == numpy.float64
        iset = handle["foo/indices"]
        assert iset.dtype == numpy.uint16
        assert list(handle["foo/shape"][:]) == [1000, 200]

    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == raw.toarray() * 10).all()

    # Trying again with a pristine object.
    out = wrap(raw)
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == raw.toarray()).all()


def test_write_sparse_matrix_Combine():
    raw1 = scipy.sparse.random(1000, 200, 0.1).tocsc()
    raw2 = scipy.sparse.random(1000, 100, 0.1).tocsc()
    out = numpy.concatenate([wrap(raw1), wrap(raw2)], axis=1)
    assert isinstance(out.seed, Combine)

    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 300), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == numpy.concatenate([raw1.toarray(), raw2.toarray()], axis=1)).all()

    # Now combining it in the other dimension.
    raw2b = scipy.sparse.random(150, 200, 0.1).tocsc()
    out = numpy.concatenate([wrap(raw1), wrap(raw2b)], axis=0)

    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1150, 200), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == numpy.concatenate([raw1.toarray(), raw2b.toarray()], axis=0)).all()


def test_write_sparse_matrix_type_guess():
    raw = scipy.sparse.random(1000, 200, 0.1)
    raw.data *= (-1)**(numpy.random.rand(len(raw.data)) > 0.5)

    # Small signed integer.
    out = numpy.round(raw * 10)
    out = out.tocsc()
    _, fpath = mkstemp(suffix=".h5")

    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.int8
    assert (numpy.array(reloaded) == out.toarray()).all()

    # Trying a larger integer.
    out = numpy.round(raw * 10000)
    out = out.tocsc()

    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.int16
    assert (numpy.array(reloaded) == out.toarray()).all()

    # Checking the unsigned choices.
    out = numpy.round(numpy.abs(raw) * 2**15)
    out = out.tocsc()

    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.uint16
    assert (numpy.array(reloaded) == out.toarray()).all()

    # Ramping up to 32 bits.
    out = numpy.round(raw * 2**30)
    out = out.tocsc()

    _, fpath = mkstemp(suffix=".h5")
    dolomite_matrix.write_sparse_matrix(out, fpath, "foo")
    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.int32
    assert (numpy.array(reloaded) == out.toarray()).all()
