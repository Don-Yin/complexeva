import numpy
import pytest

from complexeva import sample_entropy


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
