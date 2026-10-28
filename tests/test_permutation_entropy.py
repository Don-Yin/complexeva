import numpy
import pytest

from complexeva import permutation_entropy


def test_known_patterns_match_antropy():
    # antropy 0.2.3: five order-3 patterns have probabilities 2/5, 2/5, 1/5.
    x = [4, 7, 9, 10, 6, 11, 3]
    assert permutation_entropy(x) == pytest.approx(1.5219280948873621, abs=1e-14)
    assert permutation_entropy(x, normalize=True) == pytest.approx(0.588762155916294, abs=1e-14)
    assert permutation_entropy(x, order=2) == pytest.approx(0.9182958340544896, abs=1e-14)


def test_delay_changes_embedding_not_starting_step():
    assert permutation_entropy([4, 7, 9, 10, 6, 11, 3], delay=2) == pytest.approx(numpy.log2(3), abs=1e-14)


def test_ties_retain_temporal_order():
    # Stable sorting gives five different ordinal patterns for these windows.
    assert permutation_entropy([1, 1, 0, 1, 0, 0, 1]) == pytest.approx(numpy.log2(5), abs=1e-14)


@pytest.mark.parametrize("x", [numpy.arange(30), -numpy.arange(30), numpy.ones(30)])
@pytest.mark.parametrize("normalize", [False, True])
def test_monotonic_and_constant_inputs_have_zero_entropy(x, normalize):
    assert permutation_entropy(x, normalize=normalize) == 0.0


def test_uniform_patterns_normalize_to_one():
    assert permutation_entropy([0, 1, 0], order=2, normalize=True) == 1.0


def test_single_delayed_pattern_is_valid():
    assert permutation_entropy([4, 1, 7, 2, 0], order=3, delay=2) == 0.0


def test_large_order_does_not_overflow_an_integer_pattern_hash():
    x = numpy.random.default_rng(4).standard_normal(100)
    assert permutation_entropy(x, order=20) == pytest.approx(numpy.log2(81), abs=1e-14)


def test_white_noise_normalized_entropy_is_near_one():
    x = numpy.random.default_rng(0).standard_normal(8192)
    assert permutation_entropy(x, order=5, normalize=True) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("order", [1, 0, -1, 2.5, True, numpy.nan])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="order"):
        permutation_entropy(numpy.ones(20), order=order)


@pytest.mark.parametrize("delay", [0, -1, 1.5, True, numpy.nan])
def test_invalid_delay_rejected(delay):
    with pytest.raises(ValueError, match="delay"):
        permutation_entropy(numpy.ones(20), delay=delay)


@pytest.mark.parametrize("x, order, delay", [([], 3, 1), ([1, 2], 3, 1), ([1, 2, 3, 4], 3, 2)])
def test_short_series_rejected(x, order, delay):
    with pytest.raises(ValueError, match="samples"):
        permutation_entropy(x, order=order, delay=delay)


def test_numpy_integer_parameters_are_accepted():
    assert permutation_entropy(numpy.ones(20), order=numpy.int64(3), delay=numpy.int64(2)) == 0.0
