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


def _collect_from_2d_SparseNdarray(contents, fun: Callable, dtype: Callable):
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


###################################################
###################################################


@singledispatch
def collect_integer_attributes(x: Any):
    if is_sparse(x):
        collated = apply_over_blocks(x, _collect_integer_attributes_from_2d_SparseNdarray, allow_sparse=True)
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
def _collect_integer_attributes_from_2d_SparseNdarray(x: SparseNdarray) -> _IntegerAttributes:
    collected = _collect_from_2d_SparseNdarray(x.contents, _simple_integer_collector, x.dtype)
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


_FloatAttributes = namedtuple("_FloatAttributes", ["minimum", "maximum", "non_integer", "has_nan", "has_positive_inf", "has_negative_inf", "has_lowest", "has_highest", "non_zero"])


@singledispatch
def collect_float_attributes(x: Any) -> _FloatAttributes:
    if is_sparse(x):
        collated = apply_over_blocks(x, _collect_float_attributes_from_2d_SparseNdarray, allow_sparse=True)
    else:
        collated = apply_over_blocks(x, _collect_float_attributes_from_ndarray)
    return _combine_float_attributes(collated)


def _simple_float_collector(x: numpy.ndarray) -> _FloatAttributes:
    if x.size:
        stats = numpy.finfo(x.dtype)
        return _FloatAttributes(
            minimum=x.min(), 
            maximum=x.max(), 
            non_integer=(x % 1 != 0).any(),
            has_nan=numpy.isnan(x).any(),
            has_positive_inf=numpy.inf in x,
            has_negative_inf=-numpy.inf in x,
            has_lowest=stats.min in x,
            has_highest=stats.max in x,
            non_zero=0
        )
    else:
        return _FloatAttributes(
            minimum=None, 
            maximum=None, 
            non_integer=None,
            has_nan=False,
            has_positive_inf=False,
            has_negative_inf=False,
            has_lowest=False,
            has_highest=False,
            non_zero=0
        )


@collect_float_attributes.register
def _collect_float_attributes_from_ndarray(x: numpy.ndarray) -> _FloatAttributes:
    return _simple_float_collector(x)


@collect_float_attributes.register
def _collect_float_attributes_from_2d_SparseNdarray(x: SparseNdarray) -> _FloatAttributes:
    collected = _collect_from_2d_SparseNdarray(x.contents, _simple_float_collector, x.dtype)
    return _combine_float_attributes(collected)


def _combine_float_attributes(x: list[_FloatAttributes]):
    return _FloatAttributes(
        minimum=aggregate_min(x, "minimum"),
        maximum=aggregate_max(x, "maximum"),
        non_integer=aggregate_any(x, "non_integer"),
        has_nan=aggregate_any(x, "has_nan"),
        has_positive_inf=aggregate_any(x, "has_positive_inf"),
        has_negative_inf=aggregate_any(x, "has_negative_inf"),
        has_lowest=aggregate_any(x, "has_lowest"),
        has_highest=aggregate_any(x, "has_highest"),
        non_zero=0
    )


def _unique_values_from_2d_SparseNdarray(contents: SparseNdarray):
    output = set()
    if contents is not None:
        for i, node in enumerate(contents):
            output |= set(node[[2]])
    return output


def optimize_float_storage(x, missing: bool):
    attr = collect_float_attributes(x)
    lower = attr.minimum
    upper = attr.maximum

    has_missing = isinstance(x, numpy.ma.MaskedArray) # TODO: replace with a delayedarray missingness check.

    if has_missing:
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
        elif not attr.has_lowest:
            placeholder = numpy.finfo(x.dtype).min
        elif not attr.has_highest:
            placeholder = numpy.finfo(x.dtype).max

        # Fallback that just goes through and pulls out all unique values.
        # This does involve a coercion to 64-bit floats, though; that's 
        # just how 'choose_missing_float_placeholder' works currently.
        if placeholder is None:
            if is_sparse(x):
                uniq_sets = apply_over_blocks(x, _unique_values_from_2d_SparseNdarray, allow_sparse=True)
            else:
                uniq_sets = apply_over_blocks(x, lambda y : set(y))
            uniq = reduce(lambda a, b : a | b, uniq_sets)
            if numpy.ma.masked in uniq:
                uniq.remove(numpy.ma.masked)
            uniq_all = numpy.array(y for y in uniq, dtype=numpy.float64)
            copy, placeholder = dl.choose_missing_float_placeholder(uniq_all, numpy.zeros(uniq.shape[0], dtype=numpy.uint8), copy=False)
            return _OptimizedStorageParameters(type="f8", placeholder=placeholder, size=attr.non_zero)

        if x.dtype == numpy.float32:
            return _OptimizedStorageParameters(type="f4", placeholder=placeholder, size=attr.non_zero)
        else:
            return _OptimizedStorageParameters(type="f8", placeholder=placeholder, size=attr.non_zero)

    else:
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

setGeneric("collect_string_attributes", function(x) standardGeneric("collect_string_attributes"))

setMethod("collect_string_attributes", "ANY", function(x) {
    collected <- blockApply(x, function(y) {
        list(
            has_na1=any(y == "NA", na.rm=TRUE),
            has_na2=any(y == "_NA", na.rm=TRUE),
            max_len=suppressWarnings(max(nchar(y, "bytes"), na.rm=TRUE)),
            missing=anyNA(y),
            encoding=unique(Encoding(y))
        )
    })

    list(
        has_na1=aggregate_any(collected, "has_na1"),
        has_na2=aggregate_any(collected, "has_na2"),
        max_len=aggregate_max(collected, "max_len"),
        missing=aggregate_any(collected, "missing"),
        encoding=Reduce(union, lapply(collected, function(y) y$encoding))
    )
})

optimize_string_storage <- function(x) {
    attr <- collect_string_attributes(x)

    placeholder <- NULL
    if (attr$missing) {
        if (!attr$has_na1) {
            placeholder <- "NA"
        } else if (!attr$has_na2) {
            placeholder <- "_NA"
        } else {
            u <- Reduce(union, blockApply(x, function(y) unique(as.vector(y))))
            placeholder <- chooseMissingPlaceholderForHdf5(u)
        }
        attr$max_len <- max(attr$max_len, nchar(placeholder, "bytes"))
    }

    tid <- H5Tcopy("H5T_C_S1")
    H5Tset_strpad(tid, strpad = "NULLPAD")
    H5Tset_size(tid, max(1L, attr$max_len))
    if ("UTF-8" %in% attr$encoding) {
        H5Tset_cset(tid, "UTF8")
    } else {
        H5Tset_cset(tid, "ASCII")
    }

    list(type=tid, placeholder=placeholder)
}

###################################################
###################################################

setGeneric("collect_boolean_attributes", function(x) standardGeneric("collect_boolean_attributes"))

setMethod("collect_boolean_attributes", "ANY", function(x) {
    output <- list()
    if (is_sparse(x)) {
        collated <- blockApply(x, function(x) list(missing=anyNA(nzdata(x)), non_zero=length(nzdata(x))), as.sparse=TRUE)
        output$non_zero <- aggregate_sum(collated, "non_zero")
    } else {
        collated <- list(list(missing=anyNA(x)))
    }
    output$missing <- aggregate_any(collated, "missing")
    output
})

setMethod("collect_boolean_attributes", "lsparseMatrix", function(x) {
    list(missing=anyNA(x), non_zero=length(x@x))
})

setMethod("collect_boolean_attributes", "SVT_SparseMatrix", function(x) {
    collated <- collect_from_SVT(x@SVT, function(vals) { list(missing=anyNA(vals)) }, logical)
    list(
        missing=aggregate_any(collated, "missing"),
        non_zero=aggregate_sum(collated, "non_zero")
    )
})

optimize_boolean_storage <- function(x) {
    attr <- collect_boolean_attributes(x)
    if (attr$missing) {
        list(type="H5T_NATIVE_INT8", placeholder=-1L, size=attr$non_zero)
    } else {
        list(type="H5T_NATIVE_INT8", placeholder=NULL, size=attr$non_zero)
    }
}

