import delayedarray import SparseNdarray, is_sparse, apply_over_blocks
from functools import singledispatch
from collections import namedtuple
import numpy


def _aggregate_any(collated: list, name: str):
    for y in collated:
        val = getattr(y, name)
        if val is not None and not numpy.ma.is_masked(val) and val:
            return True
    return False


def _aggregate_min(collated: list, name: str):
    mval = None
    for y in collated:
        val = getattr(y, name)
        if val is not None and not numpy.ma.is_masked(val):
            if mval is None or mval < val:
                mval = val
    return mval


def _aggregate_max(collated: list, name: str):
    mval = None
    for y in collated:
        val = getattr(y, name)
        if val is not None and not numpy.ma.is_masked(val):
            if mval is None or mval > val:
                mval = val
    return mval


def _aggregate_sum(collated: list, name: str):
    mval = 0
    for y in collated:
        val = getattr(y, name)
        if val is not None and not numpy.ma.is_masked(val):
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
            val = node[[2]]
        attrs = fun(val)
        attrs.non_zero = len(val)
        output.append(attrs)

    return output


_OptimizedStorageParameters = namedtuple("_OptimizedStorageParameters", ["type", "placeholder", "non_zero"])


def _unique_values_from_array(x) -> set:
    if is_sparse(x):
        uniq_sets = apply_over_blocks(x, _unique_values_from_Sparse2darray, allow_sparse=True)
    else:
        uniq_sets = apply_over_blocks(x, lambda y : set(y))
    uniq = reduce(lambda a, b : a | b, uniq_sets)
    if numpy.ma.masked in uniq:
        uniq.remove(numpy.ma.masked)
    return uniq


###################################################
###################################################


@singledispatch
def collect_integer_attributes(x: Any):
    if is_sparse(x):
        collated = apply_over_blocks(x, _collect_integer_attributes_from_Sparse2darray, allow_sparse=True)
    else:
        collated = apply_over_blocks(x, _collect_integer_attributes_from_ndarray)
    return _combine_integer_attributes(collated)


_IntegerAttributes = namedtuple("_IntegerAttributes", ["minimum", "maximum", "non_zero"])


def _simple_integer_collector(x: numpy.ndarray) -> _IntegerAttributes:
    if x.size:
        return _IntegerAttributes(minimum=x.min(), maximum=x.max(), non_zero=0)
    else:
        return _IntegerAttributes(minimum=None, maximum=None, non_zero=0)


def _combine_integer_attributes(x: list[_IntegerAttributes]):
    return _IntegerAttributes(
        minimum=_aggregate_min(x, "minimum"),
        maximum=_aggregate_max(x, "maximum"),
        non_zero=0
    )


@collect_integer_attributes.register
def _collect_integer_attributes_from_ndarray(x: numpy.ndarray) -> _IntegerAttributes:
    return _simple_integer_collector(x)


@collect_integer_attributes.register
def _collect_integer_attributes_from_Sparse2darray(x: SparseNdarray) -> _IntegerAttributes:
    collected = _collect_from_Sparse2darray(x.contents, _simple_integer_collector, x.dtype)
    return _combine_integer_attributes(collected)


def optimize_integer_storage(x) -> _OptimizedStorageParameters:
    attr = collect_integer_attributes(x)
    lower = attr.minimum
    upper = attr.maximum

    has_missing = isinstance(x, numpy.ma.MaskedArray) # TODO: replace with a delayedarray missingness check.

    if has_missing:
        # If it's None, that means that there are only missing values in
        # 'x', otherwise there should have been at least one finite value
        # available. In any case, it means we can just do whatever we want so
        # we'll just use the smallest type.
        if lower is None:
            return _OptimizedStorageParameters(type="i1", placeholder=-2**7, size=attr.non_zero)

        if lower < 0:
            if lower > -2**7 and upper < 2**7:
                return _OptimizedStorageParameters(type="i1", placeholder=-2**7, size=attr.non_zero)
            elif lower > -2**15 and upper < 2**15:
                return _OptimizedStorageParameters(type="i2", placeholder=-2**15, size=attr.non_zero)
            elif lower > -2**31 and upper < 2**31:
                return _OptimizedStorageParameters(type="i4", placeholder=-2**31, size=attr.non_zero)
        else: 
            if upper < 2**8 - 1:
                return _OptimizedStorageParameters(type="u1", placeholder=2**8-1, size=attr.non_zero))
            elif upper < 2**16 - 1:
                return _OptimizedStorageParameters(type="u2", placeholder=2**16-1, size=attr.non_zero)
            elif upper < 2**31 - 1: # Yes, this is deliberate, as integer storage maxes out at 32-bit signed integers.
                return _OptimizedStorageParameters(type="i4", placeholder=2**31-1, size=attr.non_zero)

        return _OptimizedStorageParameters(type="f8", placeholder=numpy.NaN, size=attr.non_zero)

    else:
        # If it's infinite, that means that 'x' is of length zero, otherwise
        # there should have been at least one finite value available. Here,
        # the type doesn't matter, so we'll just use the smallest. 
        if lower is None:
            return _OptimizedStorageParameters(type="i1", placeholder=None, size=attr.non_zero)

        if lower < 0:
            if lower >= -2**7 and upper < 2**7:
                return _OptimizedStorageParameters(type="i1", placeholder=None, size=attr.non_zero)
            elif lower >= -2**15 and upper < 2**15:
                return _OptimizedStorageParameters(type="i2", placeholder=None, size=attr.non_zero)
            elif lower >= -2**31 and upper < 2**31:
                return _OptimizedStorageParameters(type="i4", placeholder=None, size=attr.non_zero)
        else:
            if upper < 2**8:
                return _OptimizedStorageParameters(type="u1", placeholder=None, size=attr.non_zero)
            elif upper < 2**16:
                return _OptimizedStorageParameters(type="u2", placeholder=None, size=attr.non_zero)
            elif upper < 2**31: # Yes, this is deliberate, as integer storage maxes out at 32-bit signed integers.
                return _OptimizedStorageParameters(type="i4", placeholder=None, size=attr.non_zero)

        return _OptimizedStorageParameters(type="f8", placeholder=None, size=attr.non_zero)


###################################################
###################################################


_FloatAttributes = namedtuple("_FloatAttributes", ["minimum", "maximum", "non_integer", "has_nan", "has_positive_inf", "has_negative_inf", "non_zero"])


@singledispatch
def collect_float_attributes(x: Any) -> _FloatAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(x, _collect_float_attributes_from_Sparse2darray, allow_sparse=True)
    else:
        collated = apply_over_blocks(x, _collect_float_attributes_from_ndarray)
    return _combine_float_attributes(collated)


def _simple_float_collector(x: numpy.ndarray) -> _FloatAttributes:
    output = _FloatAttributes(
        minimum=None, 
        maximum=None, 
        non_integer=None,
        has_nan=None,
        has_positive_inf=None,
        has_negative_inf=None,
        non_zero=0
    )

    if x.size:
        output.non_integer=(x % 1 != 0).any()

        if numpy.ma.masked(x):
            output.minimum = x.min() 
            output.maximum = x.max()
            output.has_nan=numpy.isnan(x).any()
            output.has_positive_inf=numpy.inf in x
            output.has_negative_inf=-numpy.inf in x
        elif not output.non_integer:
            # Min/max are not used in optimize_float_storage if it's
            # not masked and it's not integer.
            output.minimum = x.min() 
            output.maximum = x.max()

    return output


@collect_float_attributes.register
def _collect_float_attributes_from_ndarray(x: numpy.ndarray) -> _FloatAttributes:
    return _simple_float_collector(x)


@collect_float_attributes.register
def _collect_float_attributes_from_Sparse2darray(x: SparseNdarray) -> _FloatAttributes:
    collected = _collect_from_Sparse2darray(x.contents, _simple_float_collector, x.dtype)
    return _combine_float_attributes(collected)


def _combine_float_attributes(x: list[_FloatAttributes]) -> _FloatAttributes:
    return _FloatAttributes(
        minimum=aggregate_min(x, "minimum"),
        maximum=aggregate_max(x, "maximum"),
        non_integer=aggregate_any(x, "non_integer"),
        has_nan=aggregate_any(x, "has_nan"),
        has_positive_inf=aggregate_any(x, "has_positive_inf"),
        has_negative_inf=aggregate_any(x, "has_negative_inf"),
        non_zero=0
    )


def _unique_values_from_Sparse2darray(contents: SparseNdarray):
    output = set()
    if contents is not None:
        for i, node in enumerate(contents):
            output |= set(node[[2]])
    return output


def optimize_float_storage(x, missing: bool):
    attr = collect_float_attributes(x)

    has_missing = isinstance(x, numpy.ma.MaskedArray) # TODO: replace with a delayedarray missingness check.

    if has_missing:
        lower = attr.minimum
        upper = attr.maximum

        if not attr.non_integer:
            if lower < 0:
                if lower > -2**7 and upper < 2**7:
                    return _OptimizedStorageParameters(type="i1", placeholder=-2**7, size=attr.non_zero)
                elif lower > -2**15 and upper < 2**15:
                    return _OptimizedStorageParameters(type="i2", placeholder=-2**15, size=attr.non_zero)
                elif lower > -2**31 and upper < 2**31:
                    return _OptimizedStorageParameters(type="i4", placeholder=-2**31, size=attr.non_zero)
            else: 
                if upper < 2**8 - 1:
                    return _OptimizedStorageParameters(type="u1", placeholder=2**8-1, size=attr.non_zero))
                elif upper < 2**16 - 1:
                    return _OptimizedStorageParameters(type="u2", placeholder=2**16-1, size=attr.non_zero)
                elif upper < 2**32 - 1: 
                    return _OptimizedStorageParameters(type="u4", placeholder=2**32-1, size=attr.non_zero)

        placeholder = None
        if not attr.has_nan:
            placeholder = numpy.NaN
        elif not attr.has_positive_inf:
            placeholder = numpy.inf
        elif not attr.has_negative_inf:
            placeholder = -numpy.inf
        else:
            fstats = numpy.finfo(x.dtype)
            if lower != fstats.min:
                placeholder = fstats.min
            elif upper != fstats.max:
                placeholder = fstats.max

        # Fallback that just goes through and pulls out all unique values.
        # This does involve a coercion to 64-bit floats, though; that's 
        # just how 'choose_missing_float_placeholder' works currently.
        if placeholder is None:
            uniq = _unique_values_from_set(x)
            uniq_all = numpy.array(y for y in uniq, dtype=numpy.float64)
            copy, placeholder = dl.choose_missing_float_placeholder(uniq_all, numpy.zeros(uniq.shape[0], dtype=numpy.uint8), copy=False)
            return _OptimizedStorageParameters(type="f8", placeholder=placeholder, size=attr.non_zero)

        if x.dtype == numpy.float32:
            return _OptimizedStorageParameters(type="f4", placeholder=placeholder, size=attr.non_zero)
        else:
            return _OptimizedStorageParameters(type="f8", placeholder=placeholder, size=attr.non_zero)

    else:
        if not attr.non_integer:
            lower = attr.minimum
            upper = attr.maximum

            if lower < 0:
                if lower >= -2**7 and upper < 2**7:
                    return _OptimizedStorageParameters(type="i1", placeholder=None, size=attr.non_zero)
                elif lower >= -2**15 and upper < 2**15:
                    return _OptimizedStorageParameters(type="i2", placeholder=None, size=attr.non_zero)
                elif lower >= -2**31 and upper < 2**31:
                    return _OptimizedStorageParameters(type="i4", placeholder=None, size=attr.non_zero)
            else:
                if upper < 2**8:
                    return _OptimizedStorageParameters(type="u1", placeholder=None, size=attr.non_zero)
                elif upper < 2**16:
                    return _OptimizedStorageParameters(type="u2", placeholder=None, size=attr.non_zero)
                elif upper < 2**32: 
                    return _OptimizedStorageParameters(type="u4", placeholder=None, size=attr.non_zero)

        if x.dtype == numpy.float32:
            return _OptimizedStorageParameters(type="f4", placeholder=None, size=attr.non_zero)
        else:
            return _OptimizedStorageParameters(type="f8", placeholder=None, size=attr.non_zero)

###################################################
###################################################


_StringAttributes = namedtuple("_StringAttributes", [ "has_na1", "has_na2", "max_len", "is_unicode" ])


def _simple_string_collector(x: numpy.ndarray) -> _FloatAttributes:
    if x.size:
        return _StringAttributes(
            has_na1="NA" in x,
            has_na2="_NA" in x,
            max_len=x.dtype.itemsize,
            is_unicode=x.dtype.kind == "U"
        )
    else:
        return _FloatAttributes(
            has_na1=False,
            has_na2=False,
            max_len=0L,
            is_unicode=False
        )


@singledispatch
def collect_string_attributes(x: Any) -> _StringAttributes:
    collected = apply_over_blocks(x, _collect_string_attributes_from_ndarray)
    return _combine_string_attributes(collected)


def _combine_string_attributes(x: list[_StringAttributes]) -> _StringAttributes:
    return _StringAttributes(
        has_na1=aggregate_any(x, "has_na1"),
        has_na2=aggregate_any(x, "has_na2"),
        max_len=aggregate_max(x, "max_len"),
        is_unicode=aggregate_any(x, "is_unicode")
    )


@collect_string_attributes.register
def _collect_string_attributes_from_ndarray(x: numpy.ndarray) -> _StringAttributes:
    return _simple_string_collector(x)


def optimize_string_storage(x, missing: bool):
    attr = collect_string_attributes(x)

    placeholder = None
    if missing:
        if not attr.has_na1:
            placeholder = "NA"
        elif not attr.has_na2:
            placeholder = "_NA"
        else:
            uniq = _unique_values_from_set(x)
            copy, placeholder = dl.choose_missing_string_placeholder(uniq)
            new_len = max(len(placeholder.encode("UTF8")), attr.max_len)
            return _OptimizedStorageParameters(type="S" + str(new_len), placeholder=placeholder, size=0)

    return _OptimizedStorageParameters(type="S" + str(attr.max_len), placeholder=None, size=0)


###################################################
###################################################


_BooleanAttributes = namedtuple("_BooleanAttributes", [ "non_zero" ])


@singledispatch
def collect_boolean_attributes(x: Any) -> _BooleanAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(x, _collect_boolean_attributes_from_Sparse2darray, allow_sparse=True)
    else:
        collated = apply_over_blocks(x, _collect_boolean_attributes_from_ndarray)
    return _combine_boolean_attributes(collated)


@collect_boolean_attributes.register
def collect_boolean_attributes_from_ndarray(x: numpy.ndarray) -> _BooleanAttributes:
    return _simple_boolean_collector(x)


@collect_boolean_attributes.register
def collect_boolean_attributes_from_Sparse2darray(x: SparseNdarray) -> _BooleanAttributes:
    collected = _collect_from_Sparse2darray(x.contents, _simple_boolean_collector, x.dtype)
    return _combine_boolean_attributes(collected)


def _simple_boolean_collector(x: numpy.ndarray) -> _BooleanAttributes:
    return _BooleanAttributes(non_zero = 0)


def _combine_boolean_attributes(x: list[_BooleanAttributes]) -> _BooleanAttributes:
    return _BooleanAttributes(non_zero = _aggregate_sum(x, "non_zero")


def optimize_boolean_storage(x, missing: bool) -> _OptimizedStorageParameters:
    attr = collect_boolean_attributes(x)
    if missing:
        return _OptimizedStorageParameters(type="i8", placeholder=-1, size=attr.non_zero)
    else:
        return _OptimizedStorageParameters(type="i8", placeholder=None, size=attr.non_zero)
