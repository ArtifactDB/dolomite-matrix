from typing import Tuple, Any
import numpy
import delayedarray
import h5py


def choose_dense_chunk_sizes(shape: Tuple[int, ...], size: int, min_extent: int = 100, memory: int = 1e7) -> Tuple[int, ...]:
    """Choose some chunk sizes to use for a dense HDF5 dataset. For each
    dimension, we consider a slice of the array that consists of the full
    extent of all other dimensions. We want this slice to occupy less than
    ``memory`` in memory, and we resize the slice along the current dimension
    to achieve this. The chosen chunk size is then defined as the size of the
    slice along the current dimension. This ensures that efficient iteration
    along each dimension will not use any more than ``memory`` bytes.

    Args:
        shape: Shape of the array.

        size: Size of each array element in bytes.

        min_extent: 
            Minimum extent of each chunk dimension, to avoid problems
            with excessively small chunk sizes when the data is large.

        memory:
            Size of the (conceptual) memory buffer to use for storing blocks of
            data during iteration through the array, in bytes.

    Returns:
        Tuple containing the chunk dimensions.
    """

    num_elements = int(memory / size)
    chunks = []

    for d, s in enumerate(shape):
        otherdim = 1
        for d2, s2 in enumerate(shape): # just calculating it again to avoid overflow issues.
            if d2 != d:
                otherdim *= s2

        proposed = int(num_elements / otherdim)
        if proposed > s:
            proposed = s
        elif proposed < min_extent:
            proposed = min_extent

        chunks.append(proposed)

    return (*chunks,)


# We use a mock class 
class _DenseArrayOutputMock:
    def __init__(self, shape: Tuple, dtype: numpy.dtype, chunks: Tuple):
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks


@delayedarray.chunk_shape.register
def _chunk_shape_DenseArrayOutputMock(x: _DenseArrayOutputMock):
    return x.chunks


def _blockwise_write_to_hdf5(dhandle: h5py.Dataset, chunk_shape: Tuple, x: Any, placeholder: Any, is_string: bool, memory: int):
    mock = _DenseArrayOutputMock(x.shape, x.dtype, chunk_shape)
    block_shape = delayedarray.choose_block_shape_for_iteration(mock, memory=memory)
    if placeholder is not None:
        placeholder = x.dtype.type(placeholder)

    def _blockwise_dense_writer(pos: Tuple, block):
        if numpy.ma.is_masked(block) and block.mask.any():
            mask = block.mask
            block = block.data.copy()
            block[mask] = opts.placeholder
        if is_string:
            block = block.astype(dhandle.dtype, copy=False)
        coords = [slice(start, end) for start, end in pos]
        dhandle[*coords] = block

    delayedarray.apply_over_blocks(x, _blockwise_dense_writer, block_shape = block_shape)
    return
