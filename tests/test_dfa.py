import math
from fractions import Fraction

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


@pytest.mark.parametrize("length", [58, 59, 100, 1000, 12345])
def test_default_scales_are_floor_of_geometric_series(length):
    x = numpy.random.default_rng(length).standard_normal(length)
    max_power = int(numpy.floor(numpy.log(length / 40) / numpy.log(1.2)))
    scales = numpy.unique(numpy.floor(4 * 1.2 ** numpy.arange(max_power + 1)).astype(numpy.int64))
    assert dfa(x) == dfa(x, scales)


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


def _exact_fluctuations(x, scales):
    """RMS residual per window size with no rounding before the last two operations.

    Profile values are dyadic rationals, so prefix sums of y, m*y and y*y over a common power-of-two
    denominator are exact integers, and each window's least-squares residual is then an exact fraction.
    """
    values = numpy.asarray(x, dtype=numpy.float64)
    centered = values - values[0]
    walk = numpy.cumsum(centered - centered.mean())
    ratios = [float(v).as_integer_ratio() for v in walk]
    unit = max(den for _, den in ratios)
    sums = [[0], [0], [0]]  # prefix sums of y, m * y and y * y in units of 1 / unit
    for m, (num, den) in enumerate(ratios):
        y = num * (unit // den)
        sums[0].append(sums[0][-1] + y)
        sums[1].append(sums[1][-1] + m * y)
        sums[2].append(sums[2][-1] + y * y)
    out = []
    for size in map(int, scales):
        count = len(walk) // size
        total = Fraction(0)
        for start in range(0, count * size, size):
            stop = start + size
            level = sums[0][stop] - sums[0][start]
            cross = (sums[1][stop] - sums[1][start]) - Fraction(2 * start + size - 1, 2) * level
            total += (
                (sums[2][stop] - sums[2][start])
                - Fraction(level * level, size)
                - cross * cross * 12 / (size * (size * size - 1))
            )
        out.append(math.sqrt(float(total / (count * size) / unit**2)))
    return numpy.array(out)


def _exact_dfa(x, scales):
    """The slope that complexeva fits to the exact fluctuations."""
    scales = numpy.asarray(scales)
    return numpy.polyfit(numpy.log(scales), numpy.log(_exact_fluctuations(x, scales)), 1)[0]


def _sensitivity(scales):
    """1 / sqrt(Sxx) of the log window-size design: slope change per unit relative error in every F."""
    logs = numpy.log(numpy.asarray(scales, dtype=numpy.float64))
    return 1 / math.sqrt(numpy.sum((logs - logs.mean()) ** 2))


def _rounding_bound(scales):
    """Slope error allowed when each F is within a few ulp of exact; adjacent window sizes amplify it."""
    return 1e-13 + 64 * numpy.finfo(float).eps * _sensitivity(scales) * math.sqrt(len(scales))


def _noise(rng, n):
    return rng.standard_normal(n)


def _walk(rng, n):
    return numpy.cumsum(rng.standard_normal(n))


def _steep_trend(rng, n):
    return 5.0 * numpy.arange(n) + rng.standard_normal(n)


def _large_offset(rng, n):
    return 1e9 + rng.standard_normal(n)


def _sine(rng, n):
    return numpy.sin(numpy.arange(n) / 8) + 0.01 * rng.standard_normal(n)


SIGNALS = [_noise, _walk, _steep_trend, _large_offset, _sine]
# Default sizes for 1000 samples: floor(4 * 1.2**k) for k = 0..17, duplicates removed.
DEFAULT_1000 = numpy.unique(numpy.floor(4 * 1.2 ** numpy.arange(18)).astype(int))


@pytest.mark.parametrize("make", SIGNALS)
@pytest.mark.parametrize("scales", [DEFAULT_1000, [4, 8, 16, 64, 128], [100, 116], [5, 7, 300, 301]])
def test_slope_matches_exact_fluctuations(make, scales):
    x = make(numpy.random.default_rng(11), 1000)
    assert dfa(x, scales) == pytest.approx(_exact_dfa(x, scales), rel=0, abs=_rounding_bound(scales))


def test_default_scales_match_exact_fluctuations_to_1e_12():
    x = _walk(numpy.random.default_rng(12), 1000)
    assert dfa(x) == pytest.approx(_exact_dfa(x, DEFAULT_1000), rel=0, abs=1e-12)


@pytest.mark.parametrize(
    "seed, length, scales",
    [(5, 100000, [99998, 99999]), (6, 1000, [998, 999]), (7, 10000, [2500, 2501]), (8, 3000, [300, 301, 302])],
)
def test_adjacent_scales_stay_within_the_rounding_bound(seed, length, scales):
    # Adjacent sizes make the slope ill-conditioned; the error must stay at the level of F's rounding times that conditioning.
    x = numpy.cumsum(numpy.random.default_rng(seed).standard_normal(length))
    assert dfa(x, scales) == pytest.approx(_exact_dfa(x, scales), rel=0, abs=_rounding_bound(scales))


@pytest.mark.parametrize("dtype", [numpy.int16, numpy.int32, numpy.int64, numpy.uint16, numpy.uint64])
def test_integer_scales_and_immutable_views_match_exact_fluctuations(dtype):
    x = numpy.random.default_rng(9).standard_normal(1000)[::-1]
    x.setflags(write=False)
    scales = numpy.array([64, 16, 8, 4], dtype=dtype)
    scales.setflags(write=False)
    # Each dtype keeps its own log precision in the fit, in dfa and in the reference alike.
    assert dfa(x, scales) == pytest.approx(_exact_dfa(x, scales), rel=0, abs=_rounding_bound(scales))
