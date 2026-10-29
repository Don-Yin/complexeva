import re

import numpy
import pytest
from scipy.signal import hilbert

from complexeva import hfd, hfd_hilbert


def reference_hfd(x, k_max):
    """Literal Higuchi (1988) formula with 1-based m, written for clarity rather than speed."""
    N = len(x)
    log_inv_k, log_L = [], []
    for k in range(1, k_max + 1):
        lengths = []
        for m in range(1, k + 1):
            M = (N - m) // k
            total = sum(abs(x[m - 1 + i * k] - x[m - 1 + (i - 1) * k]) for i in range(1, M + 1))
            lengths.append(total * (N - 1) / (M * k) / k)
        log_inv_k.append(numpy.log(1.0 / k))
        log_L.append(numpy.log(numpy.mean(lengths)))
    return numpy.polyfit(log_inv_k, log_L, 1)[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [1.0, 1e120, 1e-120])
@pytest.mark.parametrize("family", ["white", "brown", "reversed"])
def test_matches_literal_formula(seed, scale, family):
    noise = numpy.random.default_rng(seed).standard_normal(500)
    x = {"white": noise, "brown": noise.cumsum(), "reversed": noise[::-1]}[family] * scale
    assert hfd(x, k_max=16) == pytest.approx(reference_hfd(x, 16), abs=1e-12)


@pytest.mark.parametrize("x", [numpy.ones(100), numpy.tile([1e308, -1e308], 20)])
def test_degenerate_lengths_give_nan(x):
    # constant input has zero length; alternating extremes overflow to inf, which fast-math must not hide
    assert numpy.isnan(hfd(x))


def test_k_max_bounds():
    x = numpy.random.default_rng(3).standard_normal(100)
    assert numpy.isfinite(hfd(x, k_max=50))
    for bad in (1, 51):
        with pytest.raises(ValueError):
            hfd(x, k_max=bad)


@pytest.mark.parametrize(
    "name, expected, tol",
    [("white", 2.0, 0.05), ("brown", 1.5, 0.08), ("sine", 1.0, 0.05)],
)
def test_known_dimensions(name, expected, tol):
    rng = numpy.random.default_rng(0)
    signals = {
        "white": rng.standard_normal(8192),
        "brown": numpy.cumsum(rng.standard_normal(8192)),
        "sine": numpy.sin(numpy.linspace(0, 20 * numpy.pi, 8192)),
    }
    assert hfd(signals[name]) == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize("measure", [hfd, hfd_hilbert])
@pytest.mark.parametrize("data", [numpy.array(1.0), numpy.ones((2, 10))])
def test_invalid_shape_names_bad_shape(measure, data):
    with pytest.raises(ValueError, match=re.escape(str(data.shape))):
        measure(data)


@pytest.mark.parametrize("measure", [hfd, hfd_hilbert])
@pytest.mark.parametrize("k_max", [1, 51, 2.5, float("nan"), "3", None])
def test_invalid_k_max_names_bad_value(measure, k_max):
    with pytest.raises(ValueError) as error:
        measure(numpy.arange(100.0), k_max=k_max)
    assert "k_max" in str(error.value)
    assert repr(k_max) in str(error.value)


@pytest.mark.parametrize("n", [20, 21, 99, 100, 257, 1000])
def test_hilbert_matches_explicit_envelope(n):
    x = numpy.random.default_rng(n).standard_normal(n)
    original = x.copy()
    x.flags.writeable = False
    assert hfd_hilbert(x) == hfd(numpy.abs(numpy.asarray(hilbert(x))))
    numpy.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("measure", [hfd, hfd_hilbert])
@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.int16])
def test_inputs_computed_in_float64(measure, dtype):
    x = (numpy.random.default_rng(8).standard_normal(100) * 100).astype(dtype)
    assert measure(x) == measure(x.astype(numpy.float64))
