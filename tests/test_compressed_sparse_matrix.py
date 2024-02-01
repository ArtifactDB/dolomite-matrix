from typing import Tuple
import scipy.sparse
import dolomite_base as dl
import dolomite_matrix as dm
from tempfile import mkdtemp
import numpy
import delayedarray
import hdf5array
import os
import random


def test_compressed_sparse_matrix_csc():
    y = scipy.sparse.random(1000, 200, 0.1).tocsc()
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype
    assert isinstance(roundtrip, dm.ReloadedArray)
    assert isinstance(roundtrip.seed.seed, hdf5array.Hdf5CompressedSparseMatrixSeed)
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


############################################
############################################


def _simulate_SparseNdarray(shape: Tuple[int, int], dtype: numpy.dtype, density: float = 0.2, mask_rate: float = 0) -> delayedarray.SparseNdarray:
    contents = []
    for i in range(shape[1]):
        all_indices = []
        for j in range(shape[0]):
            if random.random() < density:
                all_indices.append(j)

        vals =(numpy.random.rand(len(all_indices)) * 10).astype(dtype)
        if mask_rate:
            vals = numpy.ma.MaskedArray(vals, mask=numpy.random.rand(len(vals)) < mask_rate)
        contents.append((numpy.array(all_indices, dtype=numpy.dtype("int32")), vals))

    return delayedarray.SparseNdarray(shape, contents=contents)


def test_compressed_sparse_matrix_SparseNdarray_integer():
    y = _simulate_SparseNdarray((40, 35), dtype=numpy.dtype("int32"))
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)
    assert (numpy.array(roundtrip) == numpy.array(y)).all()


def test_compressed_sparse_matrix_SparseNdarray_boolean():
    y = _simulate_SparseNdarray((25, 50), dtype=numpy.dtype("bool"))
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_
    assert (numpy.array(roundtrip) == numpy.array(y)).all()


def test_compressed_sparse_matrix_SparseNdarray_float_chunks():
    y = _simulate_SparseNdarray((25, 100), dtype=numpy.dtype("float32"))
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir, compressed_sparse_matrix_buffer_size=y.dtype.itemsize * 60)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.float32
    assert (numpy.array(roundtrip) == numpy.array(y)).all()


############################################
############################################


def test_compressed_sparse_matrix_integer_mask():
    y = _simulate_SparseNdarray((50, 30), dtype=numpy.dtype("int32"), mask_rate=0.3)
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.integer)

    densed = delayedarray.to_dense_array(roundtrip)
    ref = delayedarray.to_dense_array(y)
    assert (densed.mask == ref.mask).all()
    assert numpy.logical_or(densed == ref, ref.mask).all()


def test_compressed_sparse_matrix_float_mask():
    y = _simulate_SparseNdarray((20, 100), dtype=numpy.dtype("float64"), mask_rate=0.3)
    y.contents[0] = ( # injecting some special values.
        numpy.array([1,2,3]),
        numpy.ma.MaskedArray(numpy.array([numpy.nan, numpy.inf, -numpy.inf], dtype=numpy.dtype("float64")), mask=False)
    )

    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir, compressed_sparse_matrix_buffer_size=y.dtype.itemsize * 50)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert numpy.issubdtype(roundtrip.dtype, numpy.floating)

    densed = delayedarray.to_dense_array(roundtrip)
    ref = delayedarray.to_dense_array(y)
    assert (densed.mask == ref.mask).all()
    vals = numpy.logical_or(densed == ref, numpy.isnan(densed) == numpy.isnan(ref))
    assert numpy.logical_or(vals, ref.mask).all()


def test_compressed_sparse_matrix_bool_mask():
    y = _simulate_SparseNdarray((100, 20), dtype=numpy.dtype("bool"), mask_rate=0.3)
    dir = os.path.join(mkdtemp(),"foobar")
    dl.save_object(y, dir, compressed_sparse_matrix_buffer_size=y.dtype.itemsize * 500)

    roundtrip = dl.read_object(dir)
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == numpy.bool_

    densed = delayedarray.to_dense_array(roundtrip)
    ref = delayedarray.to_dense_array(y)
    assert (densed.mask == ref.mask).all()
    assert numpy.logical_or(densed == ref, ref.mask).all()
