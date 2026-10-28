import numpy
import pytest

from complexeva import hfd


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
def test_matches_literal_formula(seed):
    x = numpy.random.default_rng(seed).standard_normal(500)
    assert hfd(x, k_max=16) == pytest.approx(reference_hfd(x, 16), abs=1e-10)


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
