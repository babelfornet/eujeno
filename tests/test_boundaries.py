import pytest
from synapse.model.blocks import compute_boundaries


def test_even_split():
    assert compute_boundaries(24, 2) == [0, 12, 24]


def test_uneven_split_is_contiguous_and_covers_all():
    b = compute_boundaries(24, 5)
    assert b[0] == 0 and b[-1] == 24
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))   # strettamente crescente
    assert len(b) == 6                                        # 5 blocchi -> 6 confini
    sizes = [b[i + 1] - b[i] for i in range(5)]
    assert max(sizes) - min(sizes) <= 1                       # il più uniforme possibile


def test_single_block():
    assert compute_boundaries(24, 1) == [0, 24]


def test_one_block_per_layer():
    assert compute_boundaries(3, 3) == [0, 1, 2, 3]


def test_rejects_too_many_blocks():
    with pytest.raises(ValueError):
        compute_boundaries(4, 5)


def test_rejects_non_positive_blocks():
    with pytest.raises(ValueError):
        compute_boundaries(24, 0)
