from numpy import issubdtype, integer, floating, bool_
from h5py import File


def _translate_array_type(dtype):
    if issubdtype(dtype, integer):
        return "integer"
    if issubdtype(dtype, floating):
        return "number"
    if dtype == bool_:
        return "boolean"
    raise NotImplementedError("staging of '" + str(type(dtype)) + "' arrays not yet supported")


def _open_writeable_hdf5_handle(path: str, cache_size: int, num_slots: int = 1000003): 
    # using a prime for the number of slots to avoid collisions in the cache.
    return File(fpath, "w", rdcc_nbytes = cache_size, rdcc_nslots = num_slots)
