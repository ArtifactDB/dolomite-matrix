from typing import Any
import numpy


def replace_mask_with_placeholder(x: numpy.ma.core.MaskedArray, placeholder: Any, output_dtype: numpy.dtype) -> numpy.ndarray:
    if placeholder is None or not numpy.ma.is_masked(x):
        return x.data.astype(output_dtype, copy=False)
    else:
        copy = x.data.astype(output_dtype, copy=True)
        copy[x.mask] = placeholder
        return copy
