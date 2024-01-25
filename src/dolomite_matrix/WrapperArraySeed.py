from typing import Tuple, Sequence, Optional
import numpy
import delayedarray


class WrapperArraySeed:
    """
    Wrapper for a DelayedArray seed, which forwards all of the required
    operations to the seed object. This is expected to be used as a base for
    concrete subclasses that attach more provenance-tracking information - see
    :py:class:`~dolomite_base.ReloadedArray.ReloadedArray` for an example.
    """

    def __init__(self, seed):
        """
        Args:
            seed: The underlying seed instance to be wrapped.
        """
        self._seed = seed

    @property
    def seed(self):
        """
        Returns:
            The underlying seed instance.
        """
        return self._seed

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            The shape of the seed.
        """
        return self._seed.shape

    @property
    def dtype(self) -> numpy.dtype:
        """
        Returns:
            The type of the seed.
        """
        return self._seed.dtype


@delayedarray.is_sparse.register
def is_sparse_WrapperArraySeed(x: WrapperArraySeed) -> bool:
    """See :py:func:`~delayedarray.is_sparse.is_sparse` for details."""
    return delayedarray.is_sparse(x._seed)


@delayedarray.chunk_shape.register
def chunk_shape_WrapperArraySeed(x: WrapperArraySeed) -> Tuple[int, ...]:
    """See :py:func:`~delayedarray.chunk_shape.chunk_shape` for details."""
    return delayedarray.chunk_shape(x._seed)


@delayedarray.extract_dense_array.register
def extract_dense_array_WrapperArraySeed(x: WrapperArraySeed, subset: Optional[Tuple[Sequence[int], ...]] = None) -> numpy.ndarray:
    """See :py:func:`~delayedarray.extract_dense_array.extract_dense_array` for details."""
    return delayedarray.extract_dense_array(x._seed, subset)


@delayedarray.extract_sparse_array.register
def extract_sparse_array_WrapperArraySeed(x: WrapperArraySeed, subset: Optional[Tuple[Sequence[int], ...]] = None) -> delayedarray.SparseNdarray:
    """See :py:func:`~delayedarray.extract_sparse_array.extract_sparse_array` for details."""
    return delayedarray.extract_sparse_array(x._seed, subset)


@delayedarray.create_dask_array.register
def create_dask_array_WrapperArraySeed(x: WrapperArraySeed):
    """See :py:func:`~delayedarray.create_dask_array.create_dask_array` for details."""
    return delayedarray.create_dask_array(x._seed)
