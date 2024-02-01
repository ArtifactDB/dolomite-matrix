from dolomite_matrix import choose_chunk_dimensions


def test_choose_chunk_dimensions():
    assert choose_chunk_dimensions((1000, 100), 4, min_extent = 0, buffer_size=8000) == (20, 2)
    assert choose_chunk_dimensions((1000, 100), 4, min_extent = 10, buffer_size=8000) == (20, 10)
    assert choose_chunk_dimensions((1000, 100), 4, min_extent = 10, buffer_size=80000) == (200, 20)
    assert choose_chunk_dimensions((1000, 100), 4, min_extent = 0, buffer_size=800000) == (1000, 100)
    assert choose_chunk_dimensions((1000, 100), 8, min_extent = 0, buffer_size=80000) == (100, 10)
    assert choose_chunk_dimensions((1000, 100), 1, min_extent = 0, buffer_size=1000) == (10, 1)


def test_choose_chunk_dimensions_3d():
    assert choose_chunk_dimensions((1000, 100, 10), 4, min_extent = 0, buffer_size=400000) == (100, 10, 1)
    assert choose_chunk_dimensions((1000, 100, 10), 1, min_extent = 0, buffer_size=400000) == (400, 40, 4)
    assert choose_chunk_dimensions((1000, 100, 10), 1, min_extent = 0, buffer_size=1e8) == (1000, 100, 10)
