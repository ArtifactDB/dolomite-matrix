from typing import Tuple, Optional, Sequence 
import delayedarray
import numpy

class DelayedMask(delayedarray.DelayedOp): 
    """
    Delayed mask to replace the missing value placeholder with a NumPy masked array.
    """

    def __init__(self, seed, placeholder, dtype: Optional[numpy.dtype] = None):
        """
        Args:
            seed:
                Any object that satisfies the seed contract,
                see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

            placeholder:
                Placeholder value for defining masked values, of the same type
                as ``seed.dtype`` (or coercible into that type). All values
                equal to the placeholder are considered to be missing.

            dtype:
                Desired type of the masked output, defaults to ``seed.dtype``.
        """
        self._seed = seed

        if numpy.issubdtype(seed.dtype, numpy.str_) and isinstance(placeholder, bytes): 
            self._placeholder = numpy.str_(placeholder.decode("UTF8"))
        else:
            self._placeholder = seed.dtype.type(placeholder)

        if dtype is None:
            dtype = seed.dtype
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of this
            object. This is the same as the ``seed`` object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> numpy.dtype:
        """
        Returns:
            NumPy type for the contents after masking.
        """
        return self._dtype

    @property
    def seed(self):
        """
        Returns:
            The seed object.
        """
        return self._seed

    @property
    def placeholder(self):
        """
        Returns:
            The placeholder value.
        """
        return self._placeholder


def _create_mask(x: numpy.ndarray, placeholder):
    if numpy.issubdtype(placeholder.dtype, numpy.floating) and numpy.isnan(placeholder):
        return numpy.isnan(x)
    else:
        return (x == placeholder)


@delayedarray.extract_dense_array.register
def extract_dense_array_DelayedMask(x: DelayedMask, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    out = delayedarray.extract_dense_array(x._seed, subset)
    mask = _create_mask(out, x._placeholder) # do this before type coercion, as the placeholder is assumed to be of the same underlying seed type.
    out = out.astype(x._dtype, copy=False)
    if mask.any():
        out = numpy.ma.MaskedArray(out, mask=mask)
    return out


def _mask_SparseNdarray(contents, placeholder, dtype):
    if not isinstance(contents, list):
        indices, values = contents
        mask = _create_mask(values, placeholder) # do this before type coercion, again.
        values = values.astype(dtype, copy=False)
        if mask.any():
            values = numpy.ma.MaskedArray(values, mask=mask)
        return indices, values
    else:
        output = []
        for val in contents:
            if val is None:
                output.append(val)
            else:
                output.append(_mask_SparseNdarray(val, placeholder, dtype))
        return output


@delayedarray.extract_sparse_array.register
def extract_sparse_array_DelayedMask(x: DelayedMask, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    out = delayedarray.extract_sparse_array(x._seed, subset)
    contents = out.contents
    if contents is not None:
        contents = _mask_SparseNdarray(contents, x._placeholder, x._dtype)
    return delayedarray.SparseNdarray(x.shape, contents, dtype=x._dtype, index_dtype=out.index_dtype, check=False)


@delayedarray.create_dask_array.register
def create_dask_array_DelayedMask(x: DelayedMask):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    target = delayedarray.create_dask_array(x._seed)
    mask = (target == x._placeholder) 
    target = target.astype(x._dtype)
    import dask
    return dask.array.ma.masked_array(target, mask=mask)


@delayedarray.chunk_shape.register
def chunk_shape_DelayedMask(x: DelayedMask):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return delayedarray.chunk_shape(x._seed)


@delayedarray.is_sparse.register
def is_sparse_DelayedMask(x: DelayedMask):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return delayedarray.is_sparse(x._seed)

