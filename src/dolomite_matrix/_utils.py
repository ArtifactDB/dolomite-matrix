from numpy import issubdtype, integer, floating, bool_, uint8, prod, ceil
from h5py import File
from typing import Tuple
from delayedarray import DelayedArray, chunk_shape


def _translate_array_type(dtype):
    if issubdtype(dtype, integer):
        return "integer"
    if issubdtype(dtype, floating):
        return "number"
    if dtype == bool_:
        return "boolean"
    raise NotImplementedError("staging of '" + str(type(dtype)) + "' arrays not yet supported")


def _choose_file_dtype(dtype):
    if dtype == bool_:
        return uint8
    return dtype


def _open_writeable_hdf5_handle(path: str, cache_size: int, num_slots: int = 1000003): 
    # using a prime for the number of slots to avoid collisions in the cache.
    return File(path, "w", rdcc_nbytes = cache_size, rdcc_nslots = num_slots)


def _choose_block_shape(x: DelayedArray, block_size: int) -> Tuple[int, ...]:
    # Block shapes are calculated by scaling up the chunks (from last to first,
    # i.e., the fastest changing to the slowest) until the block size is exceeded.
    full_shape = x.shape
    ndim = len(full_shape)

    block_shape = list(chunk_shape(x))
    block_elements = int(block_size / x.dtype.itemsize)

    for i in range(ndim - 1, -1, -1):
        current_elements = prod(block_shape) # just recompute it, avoid potential overflow issues.
        if current_elements >= block_elements:
            break
        scaling = int(block_elements / current_elements)
        if scaling == 1:
            break
        block_shape[i] = min(full_shape[i], scaling * block_shape[i])

    return (*block_shape,)


