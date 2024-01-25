import dolomite_matrix as dm
import delayedarray
import numpy


def test_DelayedMask_dense():
    y = numpy.array([[1,2,3],[4,5,6]])

    m = dm.DelayedMask(y, 1)
    assert m.dtype == y.dtype
    assert m.shape == y.shape
    assert m.placeholder == 1
    assert not delayedarray.is_sparse(m)
    assert delayedarray.chunk_shape(m) == (1, 3)

    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert numpy.ma.is_masked(block[0,0])
    assert block[1,2] == 6

    # A no-op block.
    block = delayedarray.extract_dense_array(m, (range(0, 2), range(1, 3)))
    assert not numpy.ma.is_masked(block)


def test_DelayedMask_dense_NaN():
    y = numpy.array([[1,2,numpy.NaN],[4,5,6]])
    m = dm.DelayedMask(y, numpy.NaN)
    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert numpy.ma.is_masked(block[0,2])
    assert block[0,0] == 1


def test_DelayedMask_strings():
    # Check that the placeholders work with different types of strings being compared.
    y = numpy.array(["A", "B", "C"])
    m = dm.DelayedMask(y, "C")
    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert numpy.ma.is_masked(block[2])
    assert block[0] == "A"

    m = dm.DelayedMask(y, "ASDASD")
    block = delayedarray.extract_dense_array(m)
    assert not numpy.ma.is_masked(block)
    assert m.placeholder == "ASDASD"

    m = dm.DelayedMask(y, b"B")
    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert numpy.ma.is_masked(block[1])
    assert m.placeholder == "B"

    m = dm.DelayedMask(y, numpy.bytes_("C"))
    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert m.placeholder == "C"

    # Trying with byte strings now.
    y = numpy.array([b"A", b"B", b"C"])
    m = dm.DelayedMask(y, b"A")
    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert numpy.ma.is_masked(block[0])
    assert block[2] == b"C"

    m = dm.DelayedMask(y, "ASDASD")
    block = delayedarray.extract_dense_array(m)
    assert not numpy.ma.is_masked(block)
    assert m.placeholder == b"ASDASD"

    m = dm.DelayedMask(y, numpy.str_("B"))
    block = delayedarray.extract_dense_array(m)
    assert numpy.ma.is_masked(block)
    assert m.placeholder == b"B"


def test_DelayedMask_dense_type():
    y = numpy.array([[1,2,10000],[4,5,6]])
    m = dm.DelayedMask(y, 10000, dtype=numpy.dtype(numpy.int8))

    block = delayedarray.extract_dense_array(m)
    assert block.dtype == numpy.int8
    assert numpy.ma.is_masked(block)
    assert numpy.ma.is_masked(block[0,2])
    assert block[0,0] == 1


def test_DelayedMask_sparse():
    contents = [None, (numpy.array([1, 8]), numpy.array([2, 10])), None, None, (numpy.array([0, 3, 9]), numpy.array([3, 4, 6]))]
    y = delayedarray.SparseNdarray((10, 5), contents)

    m = dm.DelayedMask(y, 4)
    assert m.dtype == y.dtype
    assert m.shape == y.shape
    assert m.placeholder == 4
    assert delayedarray.is_sparse(m)
    assert delayedarray.chunk_shape(m) == (10, 1)

    block = delayedarray.extract_sparse_array(m)
    assert block.dtype == y.dtype
    assert numpy.ma.is_masked(block.contents[4][1])
    #assert numpy.ma.is_masked(block[3,4]) # TODO: fix SparseNdarray subsetting to support masked arrays. 
    assert block[1,1] == 2


def test_DelayedMask_dense_dask():
    y = numpy.array([[1,2,10000],[4,5,6]])
    m = dm.DelayedMask(y, 10000, dtype=numpy.dtype(numpy.int8))
    da = delayedarray.create_dask_array(m).compute()
    assert numpy.ma.is_masked(da)
    assert numpy.ma.is_masked(da[0,2])
    assert da[1,2] == 6

