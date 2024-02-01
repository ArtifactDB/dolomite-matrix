from collections import namedtuple
from dataclasses import dataclass
from functools import reduce, singledispatch
from typing import Any, Callable, List, Optional, Tuple

import dolomite_base as dl
import h5py
import numpy
from delayedarray import SparseNdarray, apply_over_blocks, is_sparse, is_masked

has_scipy = False
try:
    import scipy.sparse
    has_scipy = True
except:
    pass


def _aggregate_any(collated: list, name: str):
    for y in collated:
        val = getattr(y, name)
        if val is not None and val:
            return True
    return False


def _aggregate_min(collated: list, name: str):
    mval = None
    for y in collated:
        val = getattr(y, name)
        if val is not None:
            if mval is None or mval > val:
                mval = val
    return mval


def _aggregate_max(collated: list, name: str):
    mval = None
    for y in collated:
        val = getattr(y, name)
        if val is not None:
            if mval is None or mval < val:
                mval = val
    return mval


def _aggregate_sum(collated: list, name: str):
    mval = 0
    for y in collated:
        val = getattr(y, name)
        if val is not None:
            mval += val
    return mval


def _blockwise_any(x: numpy.ndarray, condition: Callable, buffer_size: int) -> bool:
    y = x.ravel()
    step = max(1, int(buffer_size / x.dtype.itemsize))
    limit = len(y)
    for i in range(0, limit, step):
        if condition(y[i : min(limit, i + step)]).any():
            return True
    return False 


def _collect_from_Sparse2darray(contents, fun: Callable, dtype: Callable):
    if contents is None:
        attrs = fun(numpy.array([], dtype=dtype))
        attrs.non_zero = 0
        return [attrs]

    output = []
    for i, node in enumerate(contents):
        if node is None:
            val = numpy.array([], dtype=dtype)
        else:
            val = node[1]
        attrs = fun(val)
        attrs.non_zero = len(val)
        output.append(attrs)

    return output


_OptimizedStorageParameters = namedtuple("_OptimizedStorageParameters", ["type", "placeholder", "non_zero"])


def _unique_values_from_ndarray(position: Tuple, contents: numpy.ndarray) -> set:
    if not numpy.ma.is_masked(contents):
        return set(contents)
    output = set()
    for y in contents:
        if not y is numpy.ma.masked:
            output.add(y)
    return output


def _unique_values_from_Sparse2darray(position: Tuple, contents: SparseNdarray) ->set:
    output = set()
    if contents is not None:
        for i, node in enumerate(contents):
            if node is not None:
                output |= _unique_values_from_ndarray(node[[2]])
    return output


def _unique_values(x) -> set:
    if is_sparse(x):
        uniq_sets = apply_over_blocks(x, _unique_values_from_Sparse2darray, allow_sparse=True)
    else:
        uniq_sets = apply_over_blocks(x, _unique_values_from_ndarray)
    return reduce(lambda a, b : a | b, uniq_sets)


###################################################
###################################################


@dataclass
class _IntegerAttributes:
    minimum: Optional[int]
    maximum: Optional[int]
    missing: bool
    non_zero: int = 0


@singledispatch
def collect_integer_attributes(x: Any, buffer_size: int) -> _IntegerAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(
            x, 
            lambda pos, block : _collect_integer_attributes_from_Sparse2darray(block, buffer_size), 
            buffer_size = buffer_size,
            allow_sparse=True
        )
    else:
        collated = apply_over_blocks(
            x, 
            lambda pos, block : _collect_integer_attributes_from_ndarray(block, buffer_size), 
            buffer_size = buffer_size,
        )
    return _combine_integer_attributes(collated, check_missing = is_masked(x))


def _simple_integer_collector(x: numpy.ndarray, check_missing: bool) -> _IntegerAttributes:
    if x.size == 0:
        return _IntegerAttributes(minimum = None, maximum = None, missing = False)

    missing = False
    if check_missing:
        if x.mask.all():
            return _IntegerAttributes(minimum = None, maximum = None, missing = True)
        if x.mask.any():
            missing = True

    return _IntegerAttributes(minimum=x.min(), maximum=x.max(), missing=missing)


def _combine_integer_attributes(x: List[_IntegerAttributes], check_missing: bool):
    if check_missing:
        missing = _aggregate_any(x, "missing")
    else:
        missing = False

    return _IntegerAttributes(
        minimum = _aggregate_min(x, "minimum"),
        maximum = _aggregate_max(x, "maximum"),
        missing = missing,
        non_zero = _aggregate_sum(x, "non_zero"),
    )


@collect_integer_attributes.register
def _collect_integer_attributes_from_ndarray(x: numpy.ndarray, buffer_size: int) -> _IntegerAttributes:
    return _simple_integer_collector(x, check_missing = numpy.ma.isMaskedArray(x))


@collect_integer_attributes.register
def _collect_integer_attributes_from_Sparse2darray(x: SparseNdarray, buffer_size: int) -> _IntegerAttributes:
    check_missing = is_masked(x)
    collected = _collect_from_Sparse2darray(x.contents, lambda block : _simple_integer_collector(block, check_missing), x.dtype)
    return _combine_integer_attributes(collected, check_missing)


if has_scipy:
    # Currently, it seems like scipy's sparse matrices are not intended
    # to be masked, seeing as how any subsetting discards the masks, e.g.,
    #
    # >>> y = (scipy.sparse.random(1000, 200, 0.1)).tocsr()
    # >>> y.data = numpy.ma.MaskedArray(y.data, y.data > 0.5)
    # >>> y[0:5,:].data # gives back a regulary NumPy array.
    #
    # So we won't bother capturing the mask state. 

    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix, buffer_size: int):
        output = _simple_integer_collector(x.data, check_missing = False)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix, buffer_size: int):
        output = _simple_integer_collector(x.data, check_missing = False)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix, buffer_size: int):
        output = _simple_integer_collector(x.data, check_missing = False)
        output.non_zero = int(x.data.shape[0])
        return output


def optimize_integer_storage(x, buffer_size: int = 1e8) -> _OptimizedStorageParameters:
    attr = collect_integer_attributes(x, buffer_size)
    lower = attr.minimum
    upper = attr.maximum
    has_missing = attr.missing

    if has_missing:
        # If it's None, that means that there are only missing values in
        # 'x', otherwise there should have been at least one finite value
        # available. In any case, it means we can just do whatever we want so
        # we'll just use the smallest type.
        if lower is None:
            return _OptimizedStorageParameters(type="i1", placeholder=-2**7, non_zero=attr.non_zero)

        if lower < 0:
            if lower > -2**7 and upper < 2**7:
                return _OptimizedStorageParameters(type="i1", placeholder=-2**7, non_zero=attr.non_zero)
            elif lower > -2**15 and upper < 2**15:
                return _OptimizedStorageParameters(type="i2", placeholder=-2**15, non_zero=attr.non_zero)
            elif lower > -2**31 and upper < 2**31:
                return _OptimizedStorageParameters(type="i4", placeholder=-2**31, non_zero=attr.non_zero)
        else: 
            if upper < 2**8 - 1:
                return _OptimizedStorageParameters(type="u1", placeholder=2**8-1, non_zero=attr.non_zero)
            elif upper < 2**16 - 1:
                return _OptimizedStorageParameters(type="u2", placeholder=2**16-1, non_zero=attr.non_zero)
            elif upper < 2**31 - 1: # Yes, this is deliberate, as integer storage maxes out at 32-bit signed integers.
                return _OptimizedStorageParameters(type="i4", placeholder=2**31-1, non_zero=attr.non_zero)

        return _OptimizedStorageParameters(type="f8", placeholder=numpy.NaN, non_zero=attr.non_zero)

    else:
        # If it's infinite, that means that 'x' is of length zero, otherwise
        # there should have been at least one finite value available. Here,
        # the type doesn't matter, so we'll just use the smallest. 
        if lower is None:
            return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)

        if lower < 0:
            if lower >= -2**7 and upper < 2**7:
                return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)
            elif lower >= -2**15 and upper < 2**15:
                return _OptimizedStorageParameters(type="i2", placeholder=None, non_zero=attr.non_zero)
            elif lower >= -2**31 and upper < 2**31:
                return _OptimizedStorageParameters(type="i4", placeholder=None, non_zero=attr.non_zero)
        else:
            if upper < 2**8:
                return _OptimizedStorageParameters(type="u1", placeholder=None, non_zero=attr.non_zero)
            elif upper < 2**16:
                return _OptimizedStorageParameters(type="u2", placeholder=None, non_zero=attr.non_zero)
            elif upper < 2**31: # Yes, this is deliberate, as integer storage maxes out at 32-bit signed integers.
                return _OptimizedStorageParameters(type="i4", placeholder=None, non_zero=attr.non_zero)

        return _OptimizedStorageParameters(type="f8", placeholder=None, non_zero=attr.non_zero)


###################################################
###################################################


@dataclass
class _FloatAttributes:
    # Minimum and maximum are only set if non_integer = True.
    non_integer: bool
    integer_minimum: Optional[int]
    integer_maximum: Optional[int]

    # These flags are only set if check_missing = True.
    has_nan: Optional[bool]
    has_positive_inf: Optional[bool]
    has_negative_inf: Optional[bool]
    has_zero: Optional[bool]
    has_lowest: Optional[bool]
    has_highest: Optional[bool]

    missing: bool
    non_zero: int = 0


@singledispatch
def collect_float_attributes(x: Any, buffer_size: int) -> _FloatAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(
            x, 
            lambda pos, block : _collect_float_attributes_from_Sparse2darray(block, buffer_size), 
            buffer_size = buffer_size,
            allow_sparse=True
        )
    else:
        collated = apply_over_blocks(
            x, 
            lambda pos, block : _collect_float_attributes_from_ndarray(block, buffer_size), 
            buffer_size = buffer_size,
        )
    return _combine_float_attributes(collated, check_missing = is_masked(x))


def _simple_float_collector(x: numpy.ndarray, check_missing: bool, buffer_size: int) -> _FloatAttributes:
    # Do NOT set default parameters in _FloatAttributes; it's too easy to
    # forget to set one of these flags. Prefer to spell it all out explicitly
    # to avoid errors, despite the verbosity.
    if x.size == 0:
        return _FloatAttributes(
            non_integer = False,
            integer_minimum = None,
            integer_maximum = None,
            missing = False,
            has_nan = False,
            has_positive_inf = False,
            has_negative_inf = False,
            has_zero = False,
            has_lowest = False,
            has_highest = False 
        )

    missing = False
    if check_missing:
        if numpy.ma.is_masked(x):
            if x.mask.all():
                return _FloatAttributes(
                    non_integer = False,
                    integer_minimum = None,
                    integer_maximum = None,
                    missing = True,
                    has_nan = False,
                    has_positive_inf = False,
                    has_negative_inf = False,
                    has_zero = False,
                    has_lowest = False,
                    has_highest = False 
                )
            if x.mask.any():
                missing = True

        has_nan = _blockwise_any(x, numpy.isnan, buffer_size = buffer_size)
        has_positive_inf = numpy.inf in x
        has_negative_inf = -numpy.inf in x
        non_finite = (has_nan or has_positive_inf or has_negative_inf)

        fstats = numpy.finfo(x.dtype)
        has_lowest = fstats.min in x
        has_highest = fstats.max in x
        has_zero = 0 in x
    else:
        non_finite = _blockwise_any(x, lambda b : numpy.logical_not(numpy.isfinite(b)), buffer_size = buffer_size)
        has_nan = None
        has_positive_inf = None
        has_negative_inf = None
        has_lowest = None
        has_highest = None
        has_zero = None

    integer_minimum = None
    integer_maximum = None
    if non_finite:
        non_integer = True
    else:
        non_integer = _blockwise_any(x, lambda b : (b % 1 != 0), buffer_size = buffer_size)
        if not non_integer:
            integer_minimum = x.min() 
            integer_maximum = x.max()

    return _FloatAttributes(
        non_integer = non_integer,
        integer_minimum = integer_minimum,
        integer_maximum = integer_maximum,
        missing = missing,
        has_nan = has_nan,
        has_positive_inf = has_positive_inf,
        has_negative_inf = has_negative_inf,
        has_zero = has_zero,
        has_lowest = has_lowest,
        has_highest = has_highest,
    )


@collect_float_attributes.register
def _collect_float_attributes_from_ndarray(x: numpy.ndarray, buffer_size: int) -> _FloatAttributes:
    return _simple_float_collector(x, check_missing = numpy.ma.isMaskedArray(x), buffer_size = buffer_size)


@collect_float_attributes.register
def _collect_float_attributes_from_Sparse2darray(x: SparseNdarray, buffer_size: int) -> _FloatAttributes:
    check_missing = is_masked(x)
    collected = _collect_from_Sparse2darray(
        x.contents, 
        lambda block : _simple_float_collector(block, check_missing, buffer_size), 
        x.dtype
    )
    return _combine_float_attributes(collected, check_missing)


if has_scipy:
    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix, buffer_size: int) -> _FloatAttributes:
        output = _simple_float_collector(x.data, check_missing = False, buffer_size = buffer_size)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix, buffer_size: int) -> _FloatAttributes:
        output = _simple_float_collector(x.data, check_missing = False, buffer_size = buffer_size)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix, buffer_size: int) -> _FloatAttributes:
        output = _simple_float_collector(x.data, check_missing = False, buffer_size = buffer_size)
        output.non_zero = int(x.data.shape[0])
        return output


def _combine_float_attributes(x: List[_FloatAttributes], check_missing: bool) -> _FloatAttributes:
    non_integer = _aggregate_any(x, "non_integer")
    if not non_integer:
        integer_minimum = _aggregate_min(x, "integer_minimum")
        integer_maximum = _aggregate_max(x, "integer_maximum")
    else:
        integer_minimum = None
        integer_maximum = None

    if check_missing:
        missing = _aggregate_any(x, "missing")
        has_nan = _aggregate_any(x, "has_nan")
        has_positive_inf = _aggregate_any(x, "has_positive_inf")
        has_negative_inf = _aggregate_any(x, "has_negative_inf")
        has_lowest = _aggregate_any(x, "has_lowest")
        has_highest = _aggregate_any(x, "has_highest")
        has_zero = _aggregate_any(x, "has_zero")
    else:
        missing = False
        has_nan = None
        has_positive_inf = None
        has_negative_inf = None
        has_lowest = None
        has_highest = None
        has_zero = None

    return _FloatAttributes(
        non_integer = non_integer,
        integer_minimum = integer_minimum,
        integer_maximum = integer_maximum,
        missing = missing,
        has_nan = has_nan,
        has_positive_inf = has_positive_inf,
        has_negative_inf = has_negative_inf,
        has_lowest = has_lowest,
        has_highest = has_highest,
        has_zero = has_zero,
        non_zero = _aggregate_sum(x, "non_zero"),
    )


def optimize_float_storage(x, buffer_size: int = 1e8) -> _OptimizedStorageParameters:
    attr = collect_float_attributes(x, buffer_size = buffer_size)

    if attr.missing:
        if not attr.non_integer:
            lower = attr.integer_minimum
            upper = attr.integer_maximum

            # See logic in optimize_integer_storage().
            if lower is None:
                return _OptimizedStorageParameters(type="i1", placeholder=-2**7, non_zero=attr.non_zero)

            if lower < 0:
                if lower > -2**7 and upper < 2**7:
                    return _OptimizedStorageParameters(type="i1", placeholder=-2**7, non_zero=attr.non_zero)
                elif lower > -2**15 and upper < 2**15:
                    return _OptimizedStorageParameters(type="i2", placeholder=-2**15, non_zero=attr.non_zero)
                elif lower > -2**31 and upper < 2**31:
                    return _OptimizedStorageParameters(type="i4", placeholder=-2**31, non_zero=attr.non_zero)
            else: 
                if upper < 2**8 - 1:
                    return _OptimizedStorageParameters(type="u1", placeholder=2**8-1, non_zero=attr.non_zero)
                elif upper < 2**16 - 1:
                    return _OptimizedStorageParameters(type="u2", placeholder=2**16-1, non_zero=attr.non_zero)
                elif upper < 2**32 - 1: 
                    return _OptimizedStorageParameters(type="u4", placeholder=2**32-1, non_zero=attr.non_zero)

        placeholder = None
        if not attr.has_nan:
            placeholder = numpy.NaN
        elif not attr.has_positive_inf:
            placeholder = numpy.inf
        elif not attr.has_negative_inf:
            placeholder = -numpy.inf
        elif not attr.has_lowest:
            placeholder = numpy.finfo(x.dtype).min
        elif not attr.has_highest:
            placeholder = numpy.finfo(x.dtype).max
        elif not attr.has_zero:
            placeholder = 0

        # Fallback that just goes through and pulls out all unique values.
        # This does involve a coercion to 64-bit floats, though; that's 
        # just how 'choose_missing_float_placeholder' works currently.
        if placeholder is None:
            uniq = _unique_values(x)
            placeholder = dl.choose_missing_float_placeholder(uniq)
            return _OptimizedStorageParameters(type="f8", placeholder=placeholder, non_zero=attr.non_zero)

        if x.dtype == numpy.float32:
            return _OptimizedStorageParameters(type="f4", placeholder=placeholder, non_zero=attr.non_zero)
        else:
            return _OptimizedStorageParameters(type="f8", placeholder=placeholder, non_zero=attr.non_zero)

    else:
        if not attr.non_integer:
            lower = attr.integer_minimum
            upper = attr.integer_maximum

            # See logic in optimize_integer_storage().
            if lower is None:
                return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)

            if lower < 0:
                if lower >= -2**7 and upper < 2**7:
                    return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)
                elif lower >= -2**15 and upper < 2**15:
                    return _OptimizedStorageParameters(type="i2", placeholder=None, non_zero=attr.non_zero)
                elif lower >= -2**31 and upper < 2**31:
                    return _OptimizedStorageParameters(type="i4", placeholder=None, non_zero=attr.non_zero)
            else:
                if upper < 2**8:
                    return _OptimizedStorageParameters(type="u1", placeholder=None, non_zero=attr.non_zero)
                elif upper < 2**16:
                    return _OptimizedStorageParameters(type="u2", placeholder=None, non_zero=attr.non_zero)
                elif upper < 2**32: 
                    return _OptimizedStorageParameters(type="u4", placeholder=None, non_zero=attr.non_zero)

        if x.dtype == numpy.float32:
            return _OptimizedStorageParameters(type="f4", placeholder=None, non_zero=attr.non_zero)
        else:
            return _OptimizedStorageParameters(type="f8", placeholder=None, non_zero=attr.non_zero)


###################################################
###################################################


@dataclass
class _StringAttributes:
    missing: bool
    has_na1: Optional[bool]
    has_na2: Optional[bool]
    max_len: int


def _simple_string_collector(x: numpy.ndarray, check_missing: None) -> _StringAttributes:
    if x.size == 0:
        return _StringAttributes(
            missing = False,
            has_na1 = False,
            has_na2 = False,
            max_len = 0,
        )

    missing = False
    if check_missing:
        if x.mask.all():
            return _StringAttributes(
                missing = True,
                has_na1 = False,
                has_na2 = False,
                max_len = 0,
            )
        if x.mask.any():
            missing = True

    max_len = 0
    if missing:
        for y in x.ravel():
            if not numpy.ma.is_masked(y):
                candidate = len(y.encode("UTF8"))
                if max_len < candidate:
                    max_len = candidate
    else:
        max_len = max(len(y.encode("UTF8")) for y in x.ravel())

    if check_missing:
        has_na1 = x.dtype.type("NA") in x
        has_na2 = x.dtype.type("NA_") in x
    else:
        has_na1 = None
        has_na2 = None

    return _StringAttributes(
        missing = missing,
        has_na1 = has_na1,
        has_na2 = has_na2,
        max_len = max_len,
    )


@singledispatch
def collect_string_attributes(x: Any, buffer_size: int) -> _StringAttributes:
    collected = apply_over_blocks(
        x, 
        lambda pos, block : _collect_string_attributes_from_ndarray(block, buffer_size), 
        buffer_size = buffer_size,
    )
    return _combine_string_attributes(collected, check_missing = is_masked(x))


def _combine_string_attributes(x: List[_StringAttributes], check_missing: bool) -> _StringAttributes:
    if check_missing:
        missing = _aggregate_any(x, "missing")
        has_na1 = _aggregate_any(x, "has_na1")
        has_na2 = _aggregate_any(x, "has_na2")
    else:
        missing = False
        has_na1 = None
        has_na2 = None

    return _StringAttributes(
        missing = missing,
        has_na1 = has_na1,
        has_na2 = has_na2,
        max_len = _aggregate_max(x, "max_len"),
    )


@collect_string_attributes.register
def _collect_string_attributes_from_ndarray(x: numpy.ndarray, buffer_size: int) -> _StringAttributes:
    return _simple_string_collector(x, check_missing = numpy.ma.isMaskedArray(x))


def optimize_string_storage(x, buffer_size: int = 1e8) -> _OptimizedStorageParameters:
    attr = collect_string_attributes(x, buffer_size = buffer_size)
    attr.max_len = max(1, attr.max_len)

    placeholder = None
    if attr.missing:
        if not attr.has_na1:
            placeholder = "NA"
        elif not attr.has_na2:
            placeholder = "NA_"
        else:
            uniq = _unique_values(x)
            placeholder = dl.choose_missing_string_placeholder(uniq)
        attr.max_len = max(len(placeholder.encode("UTF8")), attr.max_len)

    strtype = h5py.string_dtype(encoding = "utf8", length = attr.max_len)
    return _OptimizedStorageParameters(type = strtype, placeholder = placeholder, non_zero = 0)


###################################################
###################################################


@dataclass
class _BooleanAttributes:
    missing: bool
    non_zero: int = 0


@singledispatch
def collect_boolean_attributes(x: Any, buffer_size: int) -> _BooleanAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(
            x, 
            lambda pos, block : _collect_boolean_attributes_from_Sparse2darray(block, buffer_size), 
            buffer_size = buffer_size,
            allow_sparse=True
        )
    else:
        collated = apply_over_blocks(
            x, 
            lambda pos, block : _collect_boolean_attributes_from_ndarray(block, buffer_size), 
            buffer_size = buffer_size,
        )
    return _combine_boolean_attributes(collated, check_missing = is_masked(x))


@collect_boolean_attributes.register
def _collect_boolean_attributes_from_ndarray(x: numpy.ndarray, buffer_size: int) -> _BooleanAttributes:
    return _simple_boolean_collector(x, check_missing = numpy.ma.isMaskedArray(x))


@collect_boolean_attributes.register
def _collect_boolean_attributes_from_Sparse2darray(x: SparseNdarray, buffer_size: int) -> _BooleanAttributes:
    check_missing = is_masked(x)
    collected = _collect_from_Sparse2darray(
        x.contents, 
        lambda block : _simple_boolean_collector(block, check_missing), 
        x.dtype
    )
    return _combine_boolean_attributes(collected, check_missing)


def _simple_boolean_collector(x: numpy.ndarray, check_missing: bool) -> _BooleanAttributes:
    missing = False
    if x.size:
        if check_missing:
            if x.mask.any():
                missing = True
    return _BooleanAttributes(non_zero = 0, missing = missing)


def _combine_boolean_attributes(x: List[_BooleanAttributes], check_missing: bool) -> _BooleanAttributes:
    if check_missing:
        missing = _aggregate_any(x, "missing")
    else:
        missing = False

    return _BooleanAttributes(
        missing = missing,
        non_zero = _aggregate_sum(x, "non_zero")
    )


if has_scipy:
    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix, buffer_size: int) -> _BooleanAttributes:
        output = _simple_boolean_collector(x.data, check_missing = False)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix, buffer_size: int) -> _BooleanAttributes:
        output = _simple_boolean_collector(x.data, check_missing = False)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix, buffer_size: int) -> _BooleanAttributes:
        output = _simple_boolean_collector(x.data, check_missing = False)
        output.non_zero = int(x.data.shape[0])
        return output


def optimize_boolean_storage(x, buffer_size: int = 1e8) -> _OptimizedStorageParameters:
    attr = collect_boolean_attributes(x, buffer_size)
    if attr.missing:
        return _OptimizedStorageParameters(type="i1", placeholder=-1, non_zero=attr.non_zero)
    else:
        return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)
