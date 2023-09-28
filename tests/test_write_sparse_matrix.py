import scipy.sparse
import dolomite_matrix
from tempfile import mkstemp
from filebackedarray import Hdf5CompressedSparseMatrix
import h5py
import numpy


def test_write_sparse_matrix_scipy_csc():
    out = scipy.sparse.random(1000, 200, 0.1).tocsc()
    print(type(out))
    _, fpath = mkstemp(suffix=".h5")
    saved = dolomite_matrix.write_sparse_matrix(out, fpath, "foo")

    with h5py.File(fpath, "r") as handle:
        dset = handle["foo/data"]
        assert dset.dtype == numpy.float64
        iset = handle["foo/indices"]
        assert iset.dtype == numpy.uint16
        assert list(handle["foo/shape"][:]) == [1000, 200]

    reloaded = Hdf5CompressedSparseMatrix(fpath, "foo", shape=(1000, 200), by_column=True)
    assert reloaded.dtype == numpy.float64
    assert (numpy.array(reloaded) == out.toarray()).all()
