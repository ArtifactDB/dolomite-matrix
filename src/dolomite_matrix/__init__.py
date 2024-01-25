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


from .choose_chunk_dimensions import choose_chunk_dimensions
from .save_dense_array import save_dense_array_from_ndarray 
from .read_dense_array import read_dense_array
from .save_compressed_sparse_matrix import *
from .read_compressed_sparse_matrix import read_compressed_sparse_matrix
from .save_delayed_array import save_delayed_array

from .DelayedMask import DelayedMask
from .WrapperArraySeed import WrapperArraySeed
from .ReloadedArray import ReloadedArray, ReloadedArraySeed
