from dolomite_matrix import choose_dense_chunk_sizes


def test_choose_dense_chunk_sizes():
    assert choose_dense_chunk_sizes((1000, 100), 4, min_extent = 0, memory=8000) == (20, 2)
    assert choose_dense_chunk_sizes((1000, 100), 4, min_extent = 10, memory=8000) == (20, 10)
    assert choose_dense_chunk_sizes((1000, 100), 4, min_extent = 10, memory=80000) == (200, 20)
    assert choose_dense_chunk_sizes((1000, 100), 4, min_extent = 0, memory=800000) == (1000, 100)
    assert choose_dense_chunk_sizes((1000, 100), 8, min_extent = 0, memory=80000) == (100, 10)
    assert choose_dense_chunk_sizes((1000, 100), 1, min_extent = 0, memory=1000) == (10, 1)


def test_choose_dense_chunk_sizes_3d():
    assert choose_dense_chunk_sizes((1000, 100, 10), 4, min_extent = 0, memory=400000) == (100, 10, 1)
    assert choose_dense_chunk_sizes((1000, 100, 10), 1, min_extent = 0, memory=400000) == (400, 40, 4)
    assert choose_dense_chunk_sizes((1000, 100, 10), 1, min_extent = 0, memory=1e8) == (1000, 100, 10)
