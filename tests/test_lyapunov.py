import re

import numpy
import pytest
from numba import njit

from complexeva import lyapunov_exponent
from complexeva.lyapunov_exponent import _divergence, _mean_period


def logistic(n, x0=0.3):
    x = numpy.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    return x


def henon(n):
    x, y, out = 0.1, 0.1, numpy.empty(n)
    for i in range(n):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        out[i] = x
    return out


def test_logistic_map_is_ln2():
    assert lyapunov_exponent(logistic(3000), m=1, trajectory_len=8) == pytest.approx(numpy.log(2), abs=0.03)


def test_henon_map_known_value():
    assert lyapunov_exponent(henon(3000), m=2, trajectory_len=8) == pytest.approx(0.419, abs=0.03)


def test_regular_signal_is_about_zero():
    x = numpy.sin(numpy.linspace(0, 40 * numpy.pi, 3000))
    assert abs(lyapunov_exponent(x)) < 0.01


def test_scale_and_offset_invariant():
    x = logistic(2000)
    a = lyapunov_exponent(x, m=1, trajectory_len=8)
    assert lyapunov_exponent(1000 * x + 5, m=1, trajectory_len=8) == pytest.approx(a, rel=1e-9)


def test_dt_scales_result():
    x = logistic(2000)
    assert lyapunov_exponent(x, m=1, trajectory_len=8, dt=0.5) == pytest.approx(2 * lyapunov_exponent(x, m=1, trajectory_len=8))


def test_too_short_raises():
    with pytest.raises(ValueError, match=r"len\(data\)=10.*min_tsep=5"):
        lyapunov_exponent(numpy.arange(10.0), min_tsep=5)


@pytest.mark.parametrize("data", [numpy.array(1.0), numpy.ones((2, 10))])
def test_invalid_shape_names_bad_shape(data):
    with pytest.raises(ValueError, match=re.escape(str(data.shape))):
        lyapunov_exponent(data)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("tau", 0),
        ("tau", -1),
        ("tau", 1.5),
        ("tau", True),
        ("m", 0),
        ("m", -1),
        ("m", 1.5),
        ("m", True),
        ("trajectory_len", 0),
        ("trajectory_len", 1),
        ("trajectory_len", 2.5),
        ("trajectory_len", True),
        ("fit_offset", -1),
        ("fit_offset", 19),
        ("fit_offset", 1.5),
        ("fit_offset", True),
        ("min_tsep", -1),
        ("min_tsep", 1.5),
        ("min_tsep", True),
        ("dt", 0),
        ("dt", -1.0),
        ("dt", float("nan")),
        ("dt", float("inf")),
        ("dt", True),
        ("dt", "1"),
    ],
)
def test_invalid_parameter_names_bad_value(parameter, value):
    with pytest.raises(ValueError) as error:
        lyapunov_exponent(numpy.arange(100.0), **{parameter: value})
    assert parameter in str(error.value)
    assert repr(value) in str(error.value)


def test_constant_default_window_names_bad_value():
    with pytest.raises(ValueError, match="min_tsep.*constant.*1.0"):
        lyapunov_exponent(numpy.ones(100))


def test_constant_with_explicit_window_returns_negative_infinity():
    assert lyapunov_exponent(numpy.ones(100), min_tsep=0) == -numpy.inf


def test_numpy_integer_parameters_are_valid():
    x = logistic(100)
    expected = lyapunov_exponent(x, tau=1, m=2, trajectory_len=4, fit_offset=0, min_tsep=2)
    assert (
        lyapunov_exponent(
            x,
            tau=numpy.int64(1),
            m=numpy.int64(2),
            trajectory_len=numpy.int64(4),
            fit_offset=numpy.int64(0),
            min_tsep=numpy.int64(2),
        )
        == expected
    )


@njit(cache=True)
def reference_divergence(y, window, length):
    """Preserve the adopted ascending-index scalar implementation as an oracle."""
    n = len(y) - length + 1
    neighbour = numpy.empty(n, dtype=numpy.int64)
    for i in range(n):
        best, best_j = numpy.inf, -1
        for j in range(n):
            if abs(i - j) <= window:
                continue
            d = 0.0
            for dimension in range(y.shape[1]):
                delta = y[i, dimension] - y[j, dimension]
                d += delta * delta
            if d < best:
                best, best_j = d, j
        neighbour[i] = best_j
    log_div = numpy.full(length, -numpy.inf)
    for k in range(length):
        total, count = 0.0, 0
        for i in range(n):
            d = numpy.sqrt(numpy.sum((y[i + k] - y[neighbour[i] + k]) ** 2))
            if d > 0:
                total += numpy.log(d)
                count += 1
        if count > 0:
            log_div[k] = total / count
    return log_div


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("m,tau,length,window", [(1, 1, 8, 0), (2, 1, 20, 10), (3, 2, 9, 60), (16, 2, 20, 100)])
def test_random_divergence_is_bit_identical(seed, m, tau, length, window):
    x = numpy.random.default_rng(seed).normal(size=300)
    vectors = len(x) - (m - 1) * tau
    y = numpy.stack([x[i * tau : i * tau + vectors] for i in range(m)], axis=1)
    numpy.testing.assert_array_equal(_divergence(y, window, length), reference_divergence(y, window, length))


@pytest.mark.parametrize("kind", ["constant", "quantized", "periodic", "monotone", "near-ties", "overflow", "underflow"])
@pytest.mark.parametrize("window", [0, 10, 135])
def test_degenerate_distances_and_wide_windows(kind, window):
    rng = numpy.random.default_rng(123)
    x = rng.normal(size=300)
    if kind == "constant":
        x[:] = 1
    elif kind == "quantized":
        x = rng.integers(-3, 4, size=300).astype(float)
    elif kind == "periodic":
        x = numpy.tile([0.0, 1.0, 0.0, -1.0, 2.0, 2.0], 50)
    elif kind == "monotone":
        x = numpy.arange(300.0)
    elif kind == "near-ties":
        x = rng.integers(0, 4, size=300) + 1e-14 * x
    elif kind == "overflow":
        x *= 1e200
    elif kind == "underflow":
        x *= 1e-200
    y = numpy.stack((x[:-1], x[1:]), axis=1)
    numpy.testing.assert_array_equal(_divergence(y, window, 20), reference_divergence(y, window, 20))


@pytest.mark.parametrize("kind", ["monotone", "line-noise", "random-walk", "sine-trend", "quantized", "noisy-sine"])
@pytest.mark.parametrize("m,tau", [(2, 1), (3, 2)])
@pytest.mark.parametrize("window", [0, 10, 700, 1400])
def test_block_skipping_matches_scalar_oracle(kind, m, tau, window):
    # 3000 samples give blocks of 64 sorted ranks, so whole blocks are skipped by window and by distance.
    rng = numpy.random.default_rng(7)
    t = numpy.arange(3000.0)
    x = {
        "monotone": t,
        "line-noise": t / 3000 + 1e-4 * rng.normal(size=3000),
        "random-walk": numpy.cumsum(rng.normal(size=3000)),
        "sine-trend": numpy.sin(0.2 * t) + 4 * t / 3000,
        "quantized": rng.integers(-3, 4, size=3000).astype(float),
        "noisy-sine": numpy.sin(0.2 * t) + 0.1 * rng.normal(size=3000),
    }[kind]
    vectors = len(x) - (m - 1) * tau
    y = numpy.stack([x[i * tau : i * tau + vectors] for i in range(m)], axis=1)
    numpy.testing.assert_array_equal(_divergence(y, window, 20), reference_divergence(y, window, 20))


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
@pytest.mark.parametrize("view", ["contiguous", "strided", "reversed"])
def test_public_result_preserves_fit_and_embedding(scale, view):
    x = numpy.random.default_rng(91).normal(size=512) * scale
    if view == "strided":
        x = x[::2]
    elif view == "reversed":
        x = x[::-1]
    tau, m, length, offset, dt = 3, 3, 12, 3, 0.125
    vectors = len(x) - (m - 1) * tau
    y = numpy.stack([x[i * tau : i * tau + vectors] for i in range(m)], axis=1)
    window = min(_mean_period(x), len(x) // 4)
    div = reference_divergence(y, window, length)[offset:]
    expected = float(numpy.polyfit(numpy.arange(length)[offset:], div, 1)[0] / dt)
    assert lyapunov_exponent(x, m=m, tau=tau, trajectory_len=length, fit_offset=offset, dt=dt) == expected


def test_ties_with_different_futures_and_boundary_feasibility():
    y = numpy.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [2.0, 3.0], [8.0, 5.0], [9.0, 7.0]])
    numpy.testing.assert_array_equal(_divergence(y, 1, 3), reference_divergence(y, 1, 3))
    x = numpy.arange(80.0)
    y = numpy.stack((x[:-1], x[1:]), axis=1)
    numpy.testing.assert_array_equal(_divergence(y, 29, 20), reference_divergence(y, 29, 20))
