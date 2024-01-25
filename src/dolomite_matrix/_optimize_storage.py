from collections import namedtuple
from dataclasses import dataclass
from functools import reduce, singledispatch
from typing import Any, Callable, List, Optional, Tuple

import dolomite_base as dl
import h5py
import numpy
from delayedarray import SparseNdarray, apply_over_blocks, is_sparse

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


def _blockwise_any(x: numpy.ndarray, condition: Callable):
    y = x.ravel()
    step = 100000
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
        if not numpy.ma.is_masked(y):
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


@singledispatch
def collect_integer_attributes(x: Any):
    if is_sparse(x):
        collated = apply_over_blocks(x, lambda pos, block : _collect_integer_attributes_from_Sparse2darray(block), allow_sparse=True)
    else:
        collated = apply_over_blocks(x, lambda pos, block : _collect_integer_attributes_from_ndarray(block))
    return _combine_integer_attributes(collated)


@dataclass
class _IntegerAttributes:
    minimum: Optional[int]
    maximum: Optional[int]
    missing: bool
    non_zero: int = 0


def _simple_integer_collector(x: numpy.ndarray) -> _IntegerAttributes:
    if x.size == 0:
        return _IntegerAttributes(minimum = None, maximum = None, missing = False)

    missing = False
    if numpy.ma.is_masked(x):
        if x.mask.all():
            return _IntegerAttributes(minimum = None, maximum = None, missing = True)
        if x.mask.any():
            missing = True

    return _IntegerAttributes(minimum=x.min(), maximum=x.max(), missing=missing)


def _combine_integer_attributes(x: List[_IntegerAttributes]):
    return _IntegerAttributes(
        minimum=_aggregate_min(x, "minimum"),
        maximum=_aggregate_max(x, "maximum"),
        missing=_aggregate_any(x, "missing"),
        non_zero=_aggregate_sum(x, "non_zero"),
    )


@collect_integer_attributes.register
def _collect_integer_attributes_from_ndarray(x: numpy.ndarray) -> _IntegerAttributes:
    return _simple_integer_collector(x)


@collect_integer_attributes.register
def _collect_integer_attributes_from_Sparse2darray(x: SparseNdarray) -> _IntegerAttributes:
    collected = _collect_from_Sparse2darray(x.contents, _simple_integer_collector, x.dtype)
    return _combine_integer_attributes(collected)


if has_scipy:
    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix):
        output = _simple_integer_collector(x.data)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix):
        output = _simple_integer_collector(x.data)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix):
        output = _simple_integer_collector(x.data)
        output.non_zero = int(x.data.shape[0])
        return output


def optimize_integer_storage(x) -> _OptimizedStorageParameters:
    attr = collect_integer_attributes(x)
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
    minimum: Optional[int]
    maximum: Optional[int]
    missing: bool
    non_integer: bool
    has_nan: bool
    has_positive_inf: bool
    has_negative_inf: bool
    has_zero: Optional[bool]
    has_lowest: Optional[bool]
    has_highest: Optional[bool]
    non_zero: int = 0


@singledispatch
def collect_float_attributes(x: Any, no_missing: bool) -> _FloatAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(x, lambda pos, block : _collect_float_attributes_from_Sparse2darray(block, no_missing), allow_sparse=True)
    else:
        collated = apply_over_blocks(x, lambda pos, block : _collect_float_attributes_from_ndarray(block, no_missing))
    return _combine_float_attributes(collated)


def _simple_float_collector(x: numpy.ndarray, no_missing: bool) -> _FloatAttributes:
    output = _FloatAttributes(
        minimum = None,
        maximum = None,
        missing = False,
        non_integer = False,
        has_nan = False,
        has_positive_inf = False,
        has_negative_inf = False,
        has_zero = None,
        has_lowest = None,
        has_highest = None 
    )

    if x.size == 0:
        return output

    if not no_missing:
        if numpy.ma.is_masked(x):
            if x.mask.all():
                output.missing = True
                return output
            if x.mask.any():
                output.missing = True

    # While these are technically only used if there are missing values, we
    # still need them to obtain 'non_integer', so we just compute them.
    output.has_nan = _blockwise_any(x, numpy.isnan)
    output.has_positive_inf = numpy.inf in x
    output.has_negative_inf = -numpy.inf in x

    if output.has_nan or output.has_positive_inf or output.has_negative_inf:
        output.non_integer = True
    else:
        output.non_integer = _blockwise_any(x, lambda b : (b % 1 != 0))

    # Minimum and maximum are only used if all floats contain integers.
    if not output.non_integer:
        output.minimum = x.min() 
        output.maximum = x.max()

    # Highest/lowest are only used when there might be missing values.
    if not no_missing:
        fstats = numpy.finfo(x.dtype)
        output.has_lowest = fstats.min in x
        output.has_highest = fstats.max in x
        output.has_zero = 0 in x

    return output


@collect_float_attributes.register
def _collect_float_attributes_from_ndarray(x: numpy.ndarray, no_missing: bool) -> _FloatAttributes:
    return _simple_float_collector(x, no_missing)


@collect_float_attributes.register
def _collect_float_attributes_from_Sparse2darray(x: SparseNdarray, no_missing: bool) -> _FloatAttributes:
    collected = _collect_from_Sparse2darray(x.contents, lambda block : _simple_float_collector(block, no_missing), x.dtype)
    return _combine_float_attributes(collected)


if has_scipy:
    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix, no_missing: bool):
        output = _simple_float_collector(x.data, no_missing)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix, no_missing: bool):
        output = _simple_float_collector(x.data, no_missing)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix, no_missing: bool):
        output = _simple_float_collector(x.data, no_missing)
        output.non_zero = int(x.data.shape[0])
        return output


def _combine_float_attributes(x: List[_FloatAttributes]) -> _FloatAttributes:
    return _FloatAttributes(
        minimum=_aggregate_min(x, "minimum"),
        maximum=_aggregate_max(x, "maximum"),
        non_integer=_aggregate_any(x, "non_integer"),
        missing=_aggregate_any(x, "missing"),
        has_nan=_aggregate_any(x, "has_nan"),
        has_positive_inf=_aggregate_any(x, "has_positive_inf"),
        has_negative_inf=_aggregate_any(x, "has_negative_inf"),
        has_lowest=_aggregate_any(x, "has_lowest"),
        has_highest=_aggregate_any(x, "has_highest"),
        has_zero=_aggregate_any(x, "has_zero"),
        non_zero=_aggregate_sum(x, "non_zero"),
    )


def optimize_float_storage(x) -> _OptimizedStorageParameters:
    attr = collect_float_attributes(x, isinstance(x, numpy.ndarray) and not numpy.ma.is_masked(x))

    if attr.missing:
        if not attr.non_integer:
            lower = attr.minimum
            upper = attr.maximum

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
            lower = attr.minimum
            upper = attr.maximum

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
    has_na1: bool
    has_na2: bool
    missing: bool
    max_len: int


def _simple_string_collector(x: numpy.ndarray) -> _FloatAttributes:
    if x.size == 0:
        return _StringAttributes(
            has_na1 = False,
            has_na2 = False,
            missing = False,
            max_len = 0,
        )

    missing = False
    if numpy.ma.is_masked(x):
        if x.mask.all():
            return _StringAttributes(
                has_na1=False,
                has_na2=False,
                missing=True,
                max_len=0,
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

    return _StringAttributes(
        has_na1=x.dtype.type("NA") in x,
        has_na2=x.dtype.type("NA_") in x,
        missing=missing,
        max_len=max_len,
    )


@singledispatch
def collect_string_attributes(x: Any) -> _StringAttributes:
    collected = apply_over_blocks(x, lambda pos, block : _collect_string_attributes_from_ndarray(block))
    return _combine_string_attributes(collected)


def _combine_string_attributes(x: List[_StringAttributes]) -> _StringAttributes:
    return _StringAttributes(
        has_na1 = _aggregate_any(x, "has_na1"),
        has_na2 = _aggregate_any(x, "has_na2"),
        missing = _aggregate_any(x, "missing"),
        max_len = _aggregate_max(x, "max_len"),
    )


@collect_string_attributes.register
def _collect_string_attributes_from_ndarray(x: numpy.ndarray) -> _StringAttributes:
    return _simple_string_collector(x)


def optimize_string_storage(x) -> _OptimizedStorageParameters:
    attr = collect_string_attributes(x)
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
def collect_boolean_attributes(x: Any) -> _BooleanAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(x, lambda pos, block : _collect_boolean_attributes_from_Sparse2darray(block), allow_sparse=True)
    else:
        collated = apply_over_blocks(x, lambda pos, block : _collect_boolean_attributes_from_ndarray(block))
    return _combine_boolean_attributes(collated)


@collect_boolean_attributes.register
def _collect_boolean_attributes_from_ndarray(x: numpy.ndarray) -> _BooleanAttributes:
    return _simple_boolean_collector(x)


@collect_boolean_attributes.register
def _collect_boolean_attributes_from_Sparse2darray(x: SparseNdarray) -> _BooleanAttributes:
    collected = _collect_from_Sparse2darray(x.contents, _simple_boolean_collector, x.dtype)
    return _combine_boolean_attributes(collected)


def _simple_boolean_collector(x: numpy.ndarray) -> _BooleanAttributes:
    missing = False
    if x.size:
        if numpy.ma.is_masked(x):
            if x.mask.any():
                missing = True
    return _BooleanAttributes(non_zero = 0, missing = missing)


def _combine_boolean_attributes(x: List[_BooleanAttributes]) -> _BooleanAttributes:
    return _BooleanAttributes(
        missing = _aggregate_any(x, "missing"),
        non_zero = _aggregate_sum(x, "non_zero")
    )


if has_scipy:
    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix):
        output = _simple_boolean_collector(x.data)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix):
        output = _simple_boolean_collector(x.data)
        output.non_zero = int(x.data.shape[0])
        return output


    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix):
        output = _simple_boolean_collector(x.data)
        output.non_zero = int(x.data.shape[0])
        return output


def optimize_boolean_storage(x) -> _OptimizedStorageParameters:
    attr = collect_boolean_attributes(x)
    if attr.missing:
        return _OptimizedStorageParameters(type="i1", placeholder=-1, non_zero=attr.non_zero)
    else:
        return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)
