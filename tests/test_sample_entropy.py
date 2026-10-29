import itertools

import numpy
import pytest

from complexeva import sample_entropy
from complexeva._sampen_tree import _tree_match_count
from complexeva.sample_entropy import _TREE_MAX_ORDER, _TREE_MIN_TEMPLATES, _diagonal_match_counts, _match_counts


def test_known_binary_templates_match_antropy_and_nolds():
    # antropy 0.2.3 and nolds 0.6.3, order=2, tolerance=0.5, strict Chebyshev.
    # Five length-2 pairs match, but only two still match at length 3.
    x = numpy.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    assert sample_entropy(x, tolerance=0.5) == pytest.approx(numpy.log(5 / 2), abs=1e-14)


def test_fixed_noise_matches_antropy_and_nolds():
    # antropy 0.2.3 defaults; nolds 0.6.3 with tolerance=0.2*std(x, ddof=0).
    x = numpy.random.default_rng(0).standard_normal(2048)
    assert sample_entropy(x) == pytest.approx(2.1844831289343847, abs=1e-14)


def test_minimum_length_has_one_pair():
    assert sample_entropy([1, 1, 1, 1], tolerance=0.5) == 0.0


def test_strict_tolerance_excludes_boundary_matches():
    x = [0, 0, 1, 2]
    assert numpy.isinf(sample_entropy(x, order=1, tolerance=1.0))
    assert sample_entropy(x, order=1, tolerance=numpy.nextafter(1.0, numpy.inf)) == pytest.approx(numpy.log(3 / 2))


def test_no_matching_templates_is_nan():
    assert numpy.isnan(sample_entropy(numpy.arange(10), tolerance=0.5))
    assert numpy.isnan(sample_entropy(numpy.ones(10)))
    assert numpy.isnan(sample_entropy(numpy.ones(10), tolerance=0))


def test_constant_and_periodic_signals_have_zero_entropy_at_positive_tolerance():
    assert sample_entropy(numpy.ones(10), tolerance=0.5) == 0.0
    assert sample_entropy(numpy.tile([0, 1], 32), tolerance=0.5) == 0.0


def test_matches_without_extensions_give_infinity():
    assert numpy.isinf(sample_entropy([0, 0, 1], order=1, tolerance=0.5))


def test_default_tolerance_and_scale_offset_invariance():
    x = numpy.random.default_rng(1).standard_normal(512)
    expected = sample_entropy(x, tolerance=0.2 * numpy.std(x, ddof=0))
    assert sample_entropy(x) == expected
    assert sample_entropy(7 * x + 42) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("readonly", [False, True])
def test_noncontiguous_and_readonly_inputs(readonly):
    x = numpy.random.default_rng(2).standard_normal(600)[::2]
    x.setflags(write=not readonly)
    assert sample_entropy(x) == sample_entropy(x.copy())


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_short_series_rejected(length):
    with pytest.raises(ValueError, match="samples"):
        sample_entropy(numpy.ones(length))


@pytest.mark.parametrize("order", [0, -1, 1.5, True, numpy.float64(2), numpy.nan])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="order"):
        sample_entropy(numpy.ones(10), order=order)


@pytest.mark.parametrize("tolerance", [-1, numpy.nan, numpy.inf, True, "0.2", 1j, 10**400])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        sample_entropy(numpy.ones(10), tolerance=tolerance)


def test_numpy_integer_order_is_accepted():
    assert sample_entropy(numpy.ones(10), order=numpy.int64(2), tolerance=numpy.float64(0.5)) == 0.0


def literal_counts(x, order, tolerance):
    """Evaluate the public definition independently, without spatial pruning."""
    counts = [0, 0]
    for i, j in itertools.combinations(range(len(x) - order), 2):
        if all(abs(x[i + k] - x[j + k]) < tolerance for k in range(order)):
            counts[0] += 1
            counts[1] += abs(x[i + order] - x[j + order]) < tolerance
    return tuple(counts)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [9, 17, 33, 65])
@pytest.mark.parametrize("tolerance", [numpy.nextafter(1.0, 0.0), 1.0, numpy.nextafter(1.0, numpy.inf), 3.0])
def test_spatial_counts_equal_literal_definition(order, length, tolerance):
    x = numpy.random.default_rng(length).integers(-3, 4, length).astype(float)
    expected = literal_counts(x, order, tolerance)
    actual = tuple(_tree_match_count(x, dimension, length - order, tolerance) for dimension in (order, order + 1))
    assert actual == expected
    assert _diagonal_match_counts(x, order, tolerance) == expected


@pytest.mark.parametrize("scale", [numpy.nextafter(0.0, 1.0), 1e-300, 1.0, 1e300])
def test_extreme_magnitudes_and_exact_radius(scale):
    x = numpy.tile(numpy.array([0.0, 0.0, 1.0, -1.0, 2.0, 1.0]) * scale, 7)
    for tolerance in (scale, numpy.nextafter(scale, numpy.inf)):
        for order in (1, 2, 3):
            expected = literal_counts(x, order, tolerance)
            assert _diagonal_match_counts(x, order, tolerance) == expected
            assert tuple(_tree_match_count(x, d, len(x) - order, tolerance) for d in (order, order + 1)) == expected


def test_overflowing_distances_do_not_match():
    x = numpy.tile([numpy.finfo(float).max, -numpy.finfo(float).max, 0.0], 11)
    with numpy.errstate(over="ignore"):
        expected = literal_counts(x, 2, numpy.finfo(float).max)
    assert _diagonal_match_counts(x, 2, numpy.finfo(float).max) == expected
    assert tuple(_tree_match_count(x, d, len(x) - 2, numpy.finfo(float).max) for d in (2, 3)) == expected


@pytest.mark.parametrize("order", [1, 2, _TREE_MAX_ORDER - 1, _TREE_MAX_ORDER, _TREE_MAX_ORDER + 1, 17])
def test_dispatch_boundaries_match_diagonal_reference(order):
    for n_templates in (_TREE_MIN_TEMPLATES - 1, _TREE_MIN_TEMPLATES, _TREE_MIN_TEMPLATES + 1):
        x = numpy.random.default_rng(10 + order).normal(size=n_templates + order)
        assert _match_counts(x, order, 0.2) == _diagonal_match_counts(x, order, 0.2)


def test_wide_pair_counts_exclude_self_matches():
    n, order = 100_000, 2
    x = numpy.ones(n)
    expected = (n - order) * (n - order - 1) // 2
    assert expected > numpy.iinfo(numpy.int32).max
    assert _match_counts(x, order, 0.1) == (expected, expected)
    assert sample_entropy(x, tolerance=0.1) == 0.0
    assert numpy.isnan(sample_entropy(x))


def test_tree_preserves_nan_and_infinite_results():
    assert numpy.isnan(sample_entropy(numpy.arange(4100), tolerance=0.5))
    # The only duplicate length-two prefix has a nonmatching extension.
    x = numpy.arange(4100, dtype=float)
    x[2000:2003] = [0.0, 1.0, -1.0]
    assert numpy.isinf(sample_entropy(x, tolerance=0.5))


@pytest.mark.parametrize("order", [1, 2, _TREE_MAX_ORDER])
def test_tree_matches_diagonal_on_tied_and_clustered_series(order):
    # Identical templates and tight clusters exercise whole-block counting and the leaf-pair checks.
    rng = numpy.random.default_rng(order)
    tied = rng.integers(0, 3, 2 * _TREE_MIN_TEMPLATES).astype(float)
    clustered = rng.choice([-1.0, 1.0], 2 * _TREE_MIN_TEMPLATES) + 0.05 * rng.standard_normal(2 * _TREE_MIN_TEMPLATES)
    for x, tolerance in ((tied, 1.5), (clustered, 0.2)):
        assert _match_counts(x, order, tolerance) == _diagonal_match_counts(x, order, tolerance)
