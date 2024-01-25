from typing import Tuple, Optional, Any, Union
from dolomite_base import save_object, validate_saves
import delayedarray
import numpy 
import os

from .save_compressed_sparse_matrix import _save_compressed_sparse_matrix
from .save_dense_array import _save_dense_array


@save_object.register
@validate_saves
def save_delayed_array(x: delayedarray.DelayedArray, path: str, delayed_array_preserve_operations: bool = False, **kwargs):
    """
    Method to save :py:class:`~delayedarray.DelayedArray.DelayedArray` objects
    to disk, see :py:meth:`~dolomite_base.save_object.save_object` for details. 

    If the array is pristine, we attempt to use the ``save_object`` method of
    the seed. If ``delayed_array_preserve_operations = False``, we save the
    ``DelayedArray`` as a dense array or a compressed sparse matrix.

    Args:
        x: Object to be saved.

        path: Path to a directory to save ``x``.

        delayed_array_preserve_operations:
            Whether to preserve delayed operations via the **chihaya** specification.
            Currently not supported.

        kwargs: 
            Further arguments, passed to the ``save_object`` methods for dense
            arrays and compressed sparse matrices.

    Returns:
        ``x`` is saved to ``path``.
    """
    if delayedarray.is_pristine(x):
        candidate = save_object.dispatch(type(x.seed))
        if save_object.dispatch(object) != candidate:
            return candidate(x.seed, path, delayed_array_preserve_operations=delayed_array_preserve_operations, **kwargs)

    if not delayed_array_preserve_operations:
        if delayedarray.is_sparse(x):
            return _save_compressed_sparse_matrix(x, path, **kwargs)
        return _save_dense_array(x, path, **kwargs)

    raise NotImplementedError("no support yet for delayed operations, this will be coming soon")
