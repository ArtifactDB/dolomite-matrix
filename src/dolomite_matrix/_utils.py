import numpy


def sanitize_for_writing(x, placeholder):
    if not numpy.ma.is_masked(x):
        return x
    if not x.mask.any():
        return x.data
    copy = x.data.copy()
    copy[mask] = placeholder
    return copy
