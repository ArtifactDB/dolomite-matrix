from delayedarray import SparseNdarray, is_sparse, apply_over_blocks
from typing import Callable, Any, Optional, Union, Tuple
from functools import singledispatch, reduce
from collections import namedtuple
from dataclasses import dataclass
import numpy
import dolomite_base as dl


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


def _collect_from_Sparse2darray(contents, fun: Callable, dtype: Callable):
    if contents is None:
        attrs = fun(numpy.array([0], dtype=dtype))
        attrs.non_zero = 0
        return [attrs]

    output = []
    for i, node in enumerate(contents):
        if node is None:
            val = numpy.array([0], dtype=dtype)
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


def _combine_integer_attributes(x: list[_IntegerAttributes]):
    return _IntegerAttributes(
        minimum=_aggregate_min(x, "minimum"),
        maximum=_aggregate_max(x, "maximum"),
        missing=_aggregate_any(x, "missing")
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
        return _simple_integer_collector(x.data)


    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix):
        return _simple_integer_collector(x.data)


    @collect_integer_attributes.register
    def _collect_integer_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix):
        return _simple_integer_collector(x.data)


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
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    missing: bool = False
    non_integer: bool = False 
    non_zero: int = 0


@dataclass
class _FloatPlaceholderAttributes:
    has_nan: bool
    has_positive_inf: bool
    has_negative_inf: bool
    has_zero: bool
    has_lowest: bool
    has_highest: bool


@singledispatch
def collect_float_attributes(x: Any, find_placeholders: bool) -> _FloatAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(x, lambda pos, block : _collect_float_attributes_from_Sparse2darray(block, find_placeholders), allow_sparse=True)
    else:
        collated = apply_over_blocks(x, lambda pos, block : _collect_float_attributes_from_ndarray(block, find_placeholders))
    return _combine_float_attributes(collated, find_placeholders)


def _simple_float_collector(x: numpy.ndarray, find_placeholders: bool) -> Union[_FloatAttributes, _FloatPlaceholderAttributes]:
    if find_placeholders:
        if not x.size:
            return _FloatPlaceholderAttributes(
                has_nan = False,
                has_positive_inf = False,
                has_negative_inf = False,
                has_zero = False,
                has_lowest = False,
                has_highest = False
            )

        has_nan = numpy.isnan(x).any()
        has_positive_inf = numpy.inf in x
        has_negative_inf = -numpy.inf in x
        has_zero = 0 in x

        fstats = numpy.finfo(x.dtype)
        has_lowest = fstats.min in x
        has_highest = fstats.max in x

        return _FloatPlaceholderAttributes(
            has_nan = has_nan,
            has_positive_inf = has_positive_inf,
            has_negative_inf = has_negative_inf,
            has_zero = has_zero,
            has_lowest = has_lowest,
            has_highest = has_highest
        )

    else:
        if not x.size:
            return _FloatAttributes(minimum = None, maximum = None, non_integer = False, missing = False)

        missing = False
        if numpy.ma.is_masked(x):
            if x.mask.all():
                return _FloatAttributes(minimum = None, maximum = None, non_integer = False, missing = True)
            if x.mask.any():
                missing = True

        # This evaluates to False for any of the non-finite values.
        non_integer = (x % 1 != 0).any()

        # Minimum and maximum are only used if all floats contain integers.
        minimum = None
        maximum = None
        if not non_integer:
            minimum = x.min() 
            maximum = x.max()

        return _FloatAttributes(minimum = minimum, maximum = maximum, non_integer = non_integer, missing = missing)


@collect_float_attributes.register
def _collect_float_attributes_from_ndarray(x: numpy.ndarray, find_placeholders: bool) -> _FloatAttributes:
    return _simple_float_collector(x, find_placeholders)


@collect_float_attributes.register
def _collect_float_attributes_from_Sparse2darray(x: SparseNdarray, find_placeholders: bool) -> _FloatAttributes:
    collected = _collect_from_Sparse2darray(x.contents, lambda block : _simple_float_collector(block, find_placeholders), x.dtype)
    return _combine_float_attributes(collected, find_placeholders)


if has_scipy:
    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix, find_placeholders: bool):
        return _simple_float_collector(x.data, find_placeholders)


    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix, find_placeholders: bool):
        return _simple_float_collector(x.data, find_placeholders)


    @collect_float_attributes.register
    def _collect_float_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix, find_placeholders: bool):
        return _simple_float_collector(x.data, find_placeholders)


def _combine_float_attributes(x: list[_FloatAttributes], find_placeholders: bool) -> _FloatAttributes:
    if not find_placeholders:
        return _FloatAttributes(
            minimum=_aggregate_min(x, "minimum"),
            maximum=_aggregate_max(x, "maximum"),
            non_integer=_aggregate_any(x, "non_integer"),
            missing=_aggregate_any(x, "missing")
        )
    else:
        return _FloatPlaceholderAttributes(
            has_nan=_aggregate_any(x, "has_nan"),
            has_positive_inf=_aggregate_any(x, "has_positive_inf"),
            has_negative_inf=_aggregate_any(x, "has_negative_inf"),
            has_lowest=_aggregate_any(x, "has_lowest"),
            has_highest=_aggregate_any(x, "has_highest"),
            has_zero=_aggregate_any(x, "has_zero"),
        )


def optimize_float_storage(x) -> _OptimizedStorageParameters:
    attr = collect_float_attributes(x, find_placeholders=False)

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

        attr2 = collect_float_attributes(x, find_placeholders=True)
        placeholder = None
        if not attr2.has_nan:
            placeholder = numpy.NaN
        elif not attr2.has_positive_inf:
            placeholder = numpy.inf
        elif not attr2.has_negative_inf:
            placeholder = -numpy.inf
        elif not attr2.has_lowest:
            placeholder = numpy.finfo(x.dtype).min
        elif not attr2.has_highest:
            placeholder = numpy.finfo(x.dtype).max

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
    is_unicode: bool


def _simple_string_collector(x: numpy.ndarray) -> _FloatAttributes:
    if x.size == 0:
        return _StringAttributes(
            has_na1 = False,
            has_na2 = False,
            missing = False,
            max_len = 0,
            is_unicode = False
        )

    missing = False
    if numpy.ma.is_masked(x):
        if x.mask.all():
            return _StringAttributes(
                has_na1=False,
                has_na2=False,
                missing=True,
                max_len=0,
                is_unicode=False
            )
        if x.mask.any():
            missing = True

    max_len = 0
    if missing:
        for y in x.flatten():
            if not numpy.ma.is_masked(y):
                candidate = len(y.encode("UTF8"))
                if max_len < candidate:
                    max_len = candidate
    else:
        max_len = max(len(y.encode("UTF8")) for y in x.flatten())

    return _StringAttributes(
        has_na1="NA" in x,
        has_na2="NA_" in x,
        missing=missing,
        max_len=max_len,
        is_unicode=x.dtype.kind == "U"
    )


@singledispatch
def collect_string_attributes(x: Any) -> _StringAttributes:
    collected = apply_over_blocks(x, lambda pos, block : _collect_string_attributes_from_ndarray(block))
    return _combine_string_attributes(collected)


def _combine_string_attributes(x: list[_StringAttributes]) -> _StringAttributes:
    return _StringAttributes(
        has_na1 = _aggregate_any(x, "has_na1"),
        has_na2 = _aggregate_any(x, "has_na2"),
        missing = _aggregate_any(x, "missing"),
        max_len = _aggregate_max(x, "max_len"),
        is_unicode = _aggregate_any(x, "is_unicode")
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
            new_len = max(len(placeholder.encode("UTF8")), attr.max_len)
            return _OptimizedStorageParameters(type="S" + str(new_len), placeholder=placeholder, non_zero=0)

    return _OptimizedStorageParameters(type="S" + str(attr.max_len), placeholder=placeholder, non_zero=0)


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


def _combine_boolean_attributes(x: list[_BooleanAttributes]) -> _BooleanAttributes:
    return _BooleanAttributes(
        missing = _aggregate_any(x, "missing"),
        non_zero = _aggregate_sum(x, "non_zero")
    )


if has_scipy:
    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_csc(x: scipy.sparse.csc_matrix):
        return _simple_boolean_collector(x.data)


    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_csr(x: scipy.sparse.csr_matrix):
        return _simple_boolean_collector(x.data)


    @collect_boolean_attributes.register
    def _collect_boolean_attributes_from_scipy_coo(x: scipy.sparse.coo_matrix):
        return _simple_boolean_collector(x.data)


def optimize_boolean_storage(x) -> _OptimizedStorageParameters:
    attr = collect_boolean_attributes(x)
    if attr.missing:
        return _OptimizedStorageParameters(type="i1", placeholder=-1, non_zero=attr.non_zero)
    else:
        return _OptimizedStorageParameters(type="i1", placeholder=None, non_zero=attr.non_zero)
