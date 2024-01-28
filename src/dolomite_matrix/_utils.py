import numpy


def sanitize_for_writing(x, placeholder, output_dtype):
    if not numpy.ma.isMaskedArray(x):
        return x
    if not x.mask.any():
        return x.data
    copy = x.data.astype(output_dtype, copy=True)
    copy[x.mask] = placeholder
    return copy
