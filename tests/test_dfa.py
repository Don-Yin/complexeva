import numpy
import pytest

from complexeva import dfa


def test_fixed_noise_matches_antropy():
    # antropy 0.2.3 detrended_fluctuation; its regression adds epsilon=1e-9.
    x = numpy.random.default_rng(0).standard_normal(2048)
    assert dfa(x) == pytest.approx(0.5125839794093646, abs=1e-10)


def test_custom_scales_match_nolds():
    # nolds 0.6.3 dfa: overlap=False, order=1, fit_trend=fit_exp="poly".
    x = numpy.random.default_rng(0).standard_normal(2048)
    assert dfa(x, nvals=[4, 8, 16, 32, 64, 128]) == pytest.approx(0.540949961267416, abs=1e-12)


def test_linear_signal_matches_analytic_fluctuations():
    # Integrating a linear signal gives a quadratic walk. Its detrended RMS
    # at size n is sqrt((n**2 - 1) * (n**2 - 4) / 720).
    scales = numpy.array([4, 8, 16, 32, 64, 128])
    fluctuations = numpy.sqrt((scales**2 - 1) * (scales**2 - 4) / 720)
    expected = numpy.polyfit(numpy.log(scales), numpy.log(fluctuations), 1)[0]
    assert dfa(numpy.arange(1024), nvals=scales) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("integrate, expected", [(False, 0.5), (True, 1.5)])
def test_known_scaling_exponents(integrate, expected):
    x = numpy.random.default_rng(0).standard_normal(8192)
    if integrate:
        x = numpy.cumsum(x)
    assert dfa(x) == pytest.approx(expected, abs=0.08)


def test_scale_and_offset_invariance():
    x = numpy.random.default_rng(1).standard_normal(1000)
    assert dfa(7 * x + 42) == pytest.approx(dfa(x), abs=1e-12)


def test_partial_windows_and_pooled_rms_match_literal_formula():
    x = numpy.random.default_rng(2).standard_normal(128)
    walk = numpy.cumsum(x - x.mean())
    scales = numpy.array([5, 8, 13, 21])
    fluctuations = []
    for size in scales:
        windows = walk[: len(walk) - len(walk) % size].reshape(-1, size)
        time = numpy.arange(size)
        residuals = [window - numpy.polyval(numpy.polyfit(time, window, 1), time) for window in windows]
        fluctuations.append(numpy.sqrt(numpy.mean(numpy.square(residuals))))
    expected = numpy.polyfit(numpy.log(scales), numpy.log(fluctuations), 1)[0]
    assert dfa(x, nvals=scales) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("value", [0, 0.1, 100])
def test_constant_series_is_undefined(value):
    assert numpy.isnan(dfa(numpy.full(200, value)))


def test_default_scales_need_two_distinct_sizes():
    x = numpy.random.default_rng(3).standard_normal(58)
    assert numpy.isfinite(dfa(x))
    with pytest.raises(ValueError, match="at least two distinct"):
        dfa(x[:-1])


def test_explicit_scales_support_shorter_series():
    assert numpy.isfinite(dfa(numpy.arange(6), nvals=[4, 5]))


@pytest.mark.parametrize("length", [0, 1, 5])
def test_too_short_rejected(length):
    with pytest.raises(ValueError, match="samples"):
        dfa(numpy.ones(length))


@pytest.mark.parametrize(
    "nvals",
    [[], [4], [4, 4], [4, 4, 8], [4.0, 8.0], [True, False], [3, 8], [4, 100], [[4, 8]], [[4], [8, 16]], -1, "4 8"],
)
def test_invalid_scales_rejected(nvals):
    with pytest.raises(ValueError, match="nvals"):
        dfa(numpy.arange(100), nvals=nvals)


def test_window_order_does_not_change_fit():
    x = numpy.random.default_rng(4).standard_normal(100)
    assert dfa(x, nvals=[16, 4, 8]) == pytest.approx(dfa(x, nvals=[4, 8, 16]), abs=1e-12)
