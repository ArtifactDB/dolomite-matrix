from typing import Tuple


def guess_dense_chunk_sizes(shape: Tuple[int], size: int, min_extent: int = 100, memory: int = 1e7) -> Tuple[int]:
    """Guess some chunk sizes to use for a dense HDF5 dataset. For each
    dimension, we consider a slice of the array that consists of the full
    extent of all other dimensions. We want this slice to occupy less than
    ``memory`` in memory, and we resize the slice along the current dimension
    to achieve this. The guessed chunk size is then defined as the size of the
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
