import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "dolomite-matrix"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .choose_dense_chunk_sizes import choose_dense_chunk_sizes
from .save_dense_array import save_dense_array_from_ndarray 
from .read_dense_array import read_dense_array
from .save_compressed_sparse_matrix import *
from .read_compressed_sparse_matrix import read_compressed_sparse_matrix
from .DelayedMask import DelayedMask
#from .stage_DelayedArray import stage_DelayedArray
