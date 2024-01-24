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
    y = numpy.array([], dtype=numpy.int32)
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i1"
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


def test_optimize_integer_storage_Sparse2darray():
    y = delayedarray.SparseNdarray([10, 5], None, dtype=numpy.int32, index_dtype=numpy.int8)
    opt = optim.optimize_integer_storage(y)
    assert opt.type == "i1"
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
    assert opt.non_zero == 4

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
    assert opt.non_zero == 5


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
    assert opt.non_zero == 5

    opt = optim.optimize_integer_storage(y.tocsc())
    assert opt.type == "i2"
    assert opt.placeholder is None
    assert opt.non_zero == 5

    opt = optim.optimize_integer_storage(y.tocsr())
    assert opt.type == "i2"
    assert opt.placeholder is None
    assert opt.non_zero == 5


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
    # Unsigned integers.
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

    # Signed integers.
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
    y = numpy.array([], dtype=numpy.float64)
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    # Actual floating point values.
    y = numpy.array([-1.5,2,3])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    y = numpy.array([numpy.NaN,2,3])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    y = numpy.array([numpy.inf,2,3])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None

    # 32-bit floating point values.
    y = numpy.array([-1.5,2,3], dtype=numpy.float32)
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f4"
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

    # Actual floating point values.
    y = numpy.ma.MaskedArray([-1.5,2,3], mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert numpy.isnan(opt.placeholder)

    y = numpy.ma.MaskedArray([numpy.NaN,2,3], mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder == numpy.inf

    y = numpy.ma.MaskedArray([numpy.NaN,2,numpy.inf], mask=[False, True, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder == -numpy.inf

    fstats = numpy.finfo(numpy.float64)
    y = numpy.ma.MaskedArray([numpy.NaN, 2, numpy.inf, -numpy.inf], mask=[False, True, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder == fstats.min

    y = numpy.ma.MaskedArray([numpy.NaN, 2, numpy.inf, -numpy.inf, fstats.min], mask=[False, True, False, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder == fstats.max

    y = numpy.ma.MaskedArray([numpy.NaN, 2, numpy.inf, -numpy.inf, fstats.min, fstats.max], mask=[False, True, False, False, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder == 0

    y = numpy.ma.MaskedArray([numpy.NaN, 2, numpy.inf, -numpy.inf, fstats.min, fstats.max, 0], mask=[False, True, False, False, False, False, False])
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder == fstats.min / 2 

    # 32-bit floating point values.
    y = numpy.ma.MaskedArray([-1.5,2.2,3], mask=[True, False, False], dtype=numpy.float32)
    opt = optim.optimize_float_storage(y)
    assert opt.type == "f4"
    assert numpy.isnan(opt.placeholder)


def test_optimize_float_storage_Sparse2darray():
    y = delayedarray.SparseNdarray([10, 5], None, dtype=numpy.float32, index_dtype=numpy.int8)
    opt = optim.optimize_float_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.array([1.0, 20])), 
            None, 
            (numpy.array([2, 9]), numpy.array([0, 5000.5])), 
            None
        ]
    )

    opt = optim.optimize_float_storage(y)
    assert opt.type == "f8"
    assert opt.placeholder is None
    assert opt.non_zero == 4

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.ma.MaskedArray(numpy.array([1, 2.0]), mask=True)), 
            None, 
            (numpy.array([1, 7, 9]), numpy.ma.MaskedArray(numpy.array([-1.0, -1000, 500000]), mask=False)), 
            None
        ]
    )

    opt = optim.optimize_float_storage(y)
    assert opt.type == "i4"
    assert opt.placeholder == -2**31
    assert opt.non_zero == 5


def test_optimize_float_storage_scipy():
    import scipy
    y = scipy.sparse.coo_matrix(
        (
            [1.0,-200.0,3,-4,500],
            (
                [1,2,3,4,5],
                [1,2,3,4,5]
            )
        ), 
        [10, 10]
    )
    assert y.dtype == numpy.float64

    opt = optim.optimize_float_storage(y)
    assert opt.type == "i2"
    assert opt.placeholder is None
    assert opt.non_zero == 5

    opt = optim.optimize_float_storage(y.tocsc())
    assert opt.type == "i2"
    assert opt.placeholder is None
    assert opt.non_zero == 5

    opt = optim.optimize_float_storage(y.tocsr())
    assert opt.type == "i2"
    assert opt.placeholder is None
    assert opt.non_zero == 5


def test_optimize_float_storage_Any():
    y = delayedarray.DelayedArray(numpy.array([[1,2,3],[4,5,6]]))
    y = y * 20000.000
    assert y.dtype == numpy.float64

    opt = optim.optimize_float_storage(y)
    assert opt.type == "u4"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.array([1, 2.0])), 
            None, 
            (numpy.array([2, 9]), numpy.array([0, 500.0])), 
            None
        ]
    )
    y = delayedarray.DelayedArray(y)
    opt = optim.optimize_float_storage(y * 2)
    assert opt.type == "u2"
    assert opt.placeholder is None


###################################################
###################################################


def test_optimize_string_storage_dense():
    y = numpy.array(["A","BB","CCC"])
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S3"
    assert opt.placeholder is None

    # Empty
    y = numpy.array([], dtype=numpy.str_)
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S1"
    assert opt.placeholder is None


def test_optimize_string_storage_dense_MaskedArray():
    y = numpy.ma.MaskedArray(["A","BB","CCC"], mask=[True,False,False])
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S3"
    assert opt.placeholder == "NA"

    y = numpy.ma.MaskedArray(["A","NA","CCC"], mask=[True,False,False])
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S3"
    assert opt.placeholder == "NA_"

    y = numpy.ma.MaskedArray(["A","NA","NA_","CCC"], mask=[True,False,False,False])
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S4"
    assert opt.placeholder == "NA__"

    # All masked.
    y = numpy.ma.MaskedArray(["A","BB","CCC"], mask=[True,False,False])
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S3"
    assert opt.placeholder == "NA"


def test_optimize_string_storage_Any():
    y = delayedarray.DelayedArray(numpy.array([["A","BB","CCC"],["DDDD","EEEEE","FFFFFF"]]))
    opt = optim.optimize_string_storage(y)
    assert opt.type == "S6"
    assert opt.placeholder is None


###################################################
###################################################


def test_optimize_boolean_storage_dense():
    y = numpy.array([True,False,True])
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    # Empty
    y = numpy.array([], dtype=numpy.bool_)
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None


def test_optimize_boolean_storage_dense_MaskedArray():
    # Unsigned booleans
    y = numpy.ma.MaskedArray(numpy.array([True,False,True]), mask=[True, False, False])
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -1

    # Masked but no op.
    y = numpy.ma.MaskedArray(numpy.array([True,False,True]), mask=False)
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    # Fully masked.
    y = numpy.ma.MaskedArray([True,False,False], mask=True)
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -1


def test_optimize_boolean_storage_Sparse2darray():
    y = delayedarray.SparseNdarray([10, 5], None, dtype=numpy.bool_, index_dtype=numpy.int8)
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.array([True, False])), 
            None, 
            (numpy.array([2, 9]), numpy.array([False, True])), 
            None
        ]
    )

    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None
    assert opt.non_zero == 4

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.ma.MaskedArray(numpy.array([True, False]), mask=True)), 
            None, 
            (numpy.array([1, 7, 9]), numpy.ma.MaskedArray(numpy.array([False, False, True]), mask=False)), 
            None
        ]
    )

    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder == -1
    assert opt.non_zero == 5


def test_optimize_boolean_storage_scipy():
    import scipy
    y = scipy.sparse.coo_matrix(
        (
            [True,False,True,False,True],
            (
                [1,2,3,4,5],
                [1,2,3,4,5]
            )
        ), 
        [10, 10]
    )

    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None
    assert opt.non_zero == 5

    opt = optim.optimize_boolean_storage(y.tocsc())
    assert opt.type == "i1"
    assert opt.placeholder is None
    assert opt.non_zero == 5

    opt = optim.optimize_boolean_storage(y.tocsr())
    assert opt.type == "i1"
    assert opt.placeholder is None
    assert opt.non_zero == 5


def test_optimize_boolean_storage_Any():
    y = delayedarray.DelayedArray(numpy.array([[True,False,True],[False,True,False]]))
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None

    y = delayedarray.SparseNdarray(
        [10, 5],
        [
            None, 
            (numpy.array([0, 8]), numpy.array([True, False])), 
            None, 
            (numpy.array([2, 9]), numpy.array([False, True])), 
            None
        ]
    )
    y = delayedarray.DelayedArray(y)
    opt = optim.optimize_boolean_storage(y)
    assert opt.type == "i1"
    assert opt.placeholder is None


