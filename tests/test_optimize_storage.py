import dolomite_matrix._optimize_storage as optim
import numpy
import delayedarray


###################################################
###################################################


def test_optimize_integer_storage_dense():
    # Unsigned integers
    y = numpy.array([1,2,3])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder is None

    y = numpy.array([1,2,300])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder is None

    y = numpy.array([1,2,300000])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder is None

    y = numpy.array([1,2,3000000000])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    # Signed integers
    y = numpy.array([-1,2,3])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    y = numpy.array([-1,2,200])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i2"
    assert opt.placeholder is None

    y = numpy.array([-1,2,200000])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder is None

    y = numpy.array([-1,2,-20000000000])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    # Empty
    y = numpy.array([0])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder is None


def test_optimize_integer_storage_dense_MaskedArray():
    # Unsigned integers
    y = numpy.ma.MaskedArray(numpy.array([1,2,3]), mask=[True, False, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder == 2**8 - 1

    y = numpy.ma.MaskedArray(numpy.array([1,2,300]), mask=[True, False, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder == 2**16 - 1

    y = numpy.ma.MaskedArray(numpy.array([1,2,3000000]), mask=[True, False, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder == 2**31 - 1

    y = numpy.ma.MaskedArray(numpy.array([1,2,3000000000]), mask=[True, False, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "f8"
    assert numpy.isnan(opt.placeholder)

    # Signed integers
    y = numpy.ma.MaskedArray(numpy.array([-1,2,3]), mask=[False, True, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -2**7

    y = numpy.ma.MaskedArray(numpy.array([-1,2,200]), mask=[False, True, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i2"
    assert opt.placeholder == -2**15

    y = numpy.ma.MaskedArray(numpy.array([-1,2,200000]), mask=[False, True, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder == -2**31

    y = numpy.ma.MaskedArray(numpy.array([-1,2,200000000000]), mask=[False, True, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "f8"
    assert numpy.isnan(opt.placeholder)

    # Masked large values.
    y = numpy.ma.MaskedArray(numpy.array([1000,2,3]), mask=[True, False, False])
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder == 2**8 - 1

    # Masked but no op.
    y = numpy.ma.MaskedArray(numpy.array([1000,2,3]), mask=False)
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder is None

    # Fully masked.
    y = numpy.ma.MaskedArray([1,2,3], mask=True)
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -128


def test_optimize_integer_storage_Sparse2dArray():
    y = delayedarray.SparseNdarray([10, 5], None, dtype=numpy.int32, index_dtype=numpy.int8)
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.array([1, 20])), 
            None, 
            (numpy.array([2, 9]), numpy.array([0, 5000])), 
            None
        ]
    )

    opt = optim.optimize_integer_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.ma.MaskedArray(numpy.array([1, 20]), mask=True)), 
            None, 
            (numpy.array([1, 7, 9]), numpy.ma.MaskedArray(numpy.array([-1, -1000, 500000]), mask=False)), 
            None
        ]
    )

    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder == -2**31


def test_optimize_integer_storage_scipy():
    import scipy
    y = scipy.sparse.coo_matrix(
        (
            [1,-200,3,-4,500],
            (
                [1,2,3,4,5],
                [1,2,3,4,5]
            )
        ), 
        [10, 10]
    )

    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i2"
    assert opt.placeholder is None

    opt = optim.optimize_integer_storage(y.tocsc())
    assert opt.type == "i2"
    assert opt.placeholder is None

    opt = optim.optimize_integer_storage(y.tocsr())
    assert opt.type == "i2"
    assert opt.placeholder is None


def test_optimize_integer_storage_Any():
    y = delayedarray.DelayedArray(numpy.array([[1,2,3],[4,5,6]]))
    opt = optim.optimize_integer_storage(y * 200000)
    assert opt.type == "i4"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.array([1, 20])), 
            None, 
            (numpy.array([2, 9]), numpy.array([0, 5000])), 
            None
        ]
    )
    y = delayedarray.DelayedArray(y)
    opt = optim.optimize_integer_storage(y * 2)
    assert opt.type == "u2"
    assert opt.placeholder is None


###################################################
###################################################


def test_optimize_float_storage_dense():
    # Unsigned floats
    y = numpy.array([1.0,2,3])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder is None

    y = numpy.array([1.0,2,300])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder is None

    y = numpy.array([1.0,2,300000])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u4"
    assert opt.placeholder is None

    y = numpy.array([1.0,2,30000000000])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    # Signed floats
    y = numpy.array([-1.0,2,3])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    y = numpy.array([-1.0,2,200])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i2"
    assert opt.placeholder is None

    y = numpy.array([-1.0,2,200000])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder is None

    y = numpy.array([-1.0,2,-20000000000])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    # Empty
    y = numpy.array([0])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder is None


def test_optimize_float_storage_dense_MaskedArray():
    # Unsigned floats
    y = numpy.ma.MaskedArray(numpy.array([1.0,2,3]), mask=[True, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder == 2**8 - 1

    y = numpy.ma.MaskedArray(numpy.array([1.0,2,300]), mask=[True, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder == 2**16 - 1

    y = numpy.ma.MaskedArray(numpy.array([1.0,2,3000000]), mask=[True, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u4"
    assert opt.placeholder == 2**32 - 1

    y = numpy.ma.MaskedArray(numpy.array([1.0,2,30000000000]), mask=[True, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert numpy.isnan(opt.placeholder)

    # Signed floats
    y = numpy.ma.MaskedArray(numpy.array([-1.0,2,3]), mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -2**7

    y = numpy.ma.MaskedArray(numpy.array([-1.0,2,200]), mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i2"
    assert opt.placeholder == -2**15

    y = numpy.ma.MaskedArray(numpy.array([-1.0,2,200000]), mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder == -2**31

    y = numpy.ma.MaskedArray(numpy.array([-1.0,2,200000000000]), mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert numpy.isnan(opt.placeholder)

    # Masked large values.
    y = numpy.ma.MaskedArray(numpy.array([1000.0,2,3]), mask=[True, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u1"
    assert opt.placeholder == 2**8 - 1

    # Masked but no op.
    y = numpy.ma.MaskedArray(numpy.array([1000.0,2,3]), mask=False)
    opt = optim.optimize_float_storage(y)
    assert opt.type == "u2"
    assert opt.placeholder is None

    # Fully masked.
    y = numpy.ma.MaskedArray([1.0,2,3], mask=True)
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -128


