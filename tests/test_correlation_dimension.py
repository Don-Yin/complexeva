"""Literal-formula, reference-package and batch-contract tests for correlation dimension."""

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import pdist

from complexeva.correlation_dimension import _correlation_counts, correlation_dimension


def literal(x, order=2, delay=1, radii=None):
    """Small dense reference: distinct unordered Euclidean pairs with strict radii."""
    x = np.asarray(x, dtype=np.float64)
    if radii is None:
        radii = np.std(x, ddof=1) * 0.1 * 1.03 ** np.arange(55)
    radii = np.asarray(radii)
    vectors = sliding_window_view(x, (order - 1) * delay + 1)[:, ::delay]
    distances = pdist(vectors, "euclidean")
    sums = np.array([np.count_nonzero(distances < r) for r in radii]) / len(distances)
    keep = sums > 0
    if keep.sum() < 2:
        return np.nan
    return np.polyfit(np.log(radii[keep]), np.log(sums[keep]), 1)[0]


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("delay", [1, 2, 4])
@pytest.mark.parametrize("kind", ["normal", "binary", "trend", "walk"])
def test_literal_formula(order, delay, kind):
    rng = np.random.default_rng(42)
    x = rng.normal(size=128)
    if kind == "binary":
        x = (x > 0).astype(float)
    elif kind == "trend":
        x += np.linspace(0, 10, len(x))
    elif kind == "walk":
        x = x.cumsum()
    radii = np.geomspace(0.1, 5, 23)
    assert correlation_dimension(x, order, delay, radii) == pytest.approx(literal(x, order, delay, radii), abs=2e-14)


def test_strict_boundaries_and_duplicate_points():
    points = np.array([[0.0], [1.0], [1.0], [2.0], [4.0], [5.0]])
    radii = np.array([np.nextafter(1.0, 0), 1.0, np.nextafter(1.0, np.inf), 2.0, 5.0])
    expected = np.array([2 * np.count_nonzero(pdist(points) < r) for r in radii])
    np.testing.assert_array_equal(_correlation_counts(points, radii), expected)
    assert correlation_dimension(points[:, 0], order=1, radii=radii) == pytest.approx(literal(points[:, 0], 1, radii=radii))


@pytest.mark.parametrize("order", [1, 2, 3, 5, 7, 8])
def test_irrational_distance_boundaries(order):
    x = np.arange(order + 1, dtype=float)
    distance = np.sqrt(order)
    radii = np.array([np.nextafter(distance, 0), distance, np.nextafter(distance, np.inf), 2 * distance])
    points = sliding_window_view(x, order)
    expected = np.array([2 * int(distance < r) for r in radii])
    np.testing.assert_array_equal(_correlation_counts(points, radii), expected)
    assert correlation_dimension(x, order, radii=radii) == pytest.approx(literal(x, order, radii=radii), abs=2e-14)


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_all_integer_point_distance_boundaries(dimension):
    points = np.random.default_rng(8).integers(0, 5, size=(32, dimension)).astype(float)
    distances = pdist(points)
    radii = np.unique(np.concatenate([np.nextafter(distances, 0), distances, np.nextafter(distances, np.inf)]))
    radii = radii[radii > 0]
    expected = np.array([2 * np.count_nonzero(distances < r) for r in radii])
    np.testing.assert_array_equal(_correlation_counts(points, radii), expected)


@pytest.mark.parametrize("order,delay", [(1, 1), (2, 1), (3, 2)])
def test_neurokit2_matched_radii(order, delay):
    nk = pytest.importorskip("neurokit2")
    x = np.random.default_rng(13).normal(size=400)
    expected, _ = nk.fractal_correlation(x, dimension=order, delay=delay, radius="nolds")
    assert correlation_dimension(x, order, delay) == pytest.approx(expected, abs=2e-13)
    expected, info = nk.fractal_correlation(x, dimension=order, delay=delay)
    assert correlation_dimension(x, order, delay, info["Radius"]) == pytest.approx(expected, abs=2e-13)


def test_nolds_diagonal_convention():
    nolds = pytest.importorskip("nolds")
    x = np.random.default_rng(19).normal(size=500)
    raw, (log_r, log_c, _) = nolds.corr_dim(x, emb_dim=2, fit="poly", debug_data=True)
    m = len(x) - 1
    # nolds' zero distance-matrix diagonal contributes m spurious matches.
    corrected_sums = np.exp(log_c) - 1 / (m - 1)
    corrected = np.polyfit(log_r, np.log(corrected_sums), 1)[0]
    actual = correlation_dimension(x)
    assert actual == pytest.approx(corrected, abs=2e-13)
    assert abs(actual - raw) > 0.01


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "reverse", "stride", "transpose", "int", "float32"])
@pytest.mark.parametrize("radii", [None, np.geomspace(0.03, 3, 18)])
def test_batch_bitwise(layout, radii):
    x = np.random.default_rng(1).normal(size=(2, 3, 101))
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[..., ::-1]
    elif layout == "stride":
        x = x[..., ::2]
    elif layout == "transpose":
        x = x.transpose(1, 0, 2)
    elif layout == "int":
        x = (x * 10).astype(np.int16)
    elif layout == "float32":
        x = x.astype(np.float32)
    actual = correlation_dimension(x, order=3, delay=2, radii=radii)
    expected = np.array([correlation_dimension(x[i], 3, 2, radii) for i in np.ndindex(x.shape[:-1])]).reshape(x.shape[:-1])
    assert isinstance(actual, np.ndarray)
    assert actual.shape == x.shape[:-1]
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_degenerate_and_empty_batch():
    assert np.isnan(correlation_dimension(np.ones(20)))
    assert np.isnan(correlation_dimension(np.ones(20), radii=[0.1, 0.2]))
    assert np.isnan(correlation_dimension(np.arange(20), radii=[1e-6, 1e-5]))
    assert np.isnan(correlation_dimension(np.arange(4), order=1, radii=[0.1, 0.2, 2.0]))
    empty = correlation_dimension(np.empty((2, 0, 30)))
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (2, 0)
    out = correlation_dimension(np.stack([np.ones(20), np.arange(20)]))
    assert isinstance(out, np.ndarray)
    assert np.isnan(out[0])


@pytest.mark.parametrize("factor", [1e-250, 1e-120, 1e120, 1e250])
def test_extreme_scale_invariance(factor):
    x = np.random.default_rng(7).normal(size=400)
    assert correlation_dimension(x * factor) == pytest.approx(correlation_dimension(x), abs=2e-13)
    radii = np.geomspace(0.05, 0.5, 20)
    assert correlation_dimension(x * factor, radii=radii * factor) == pytest.approx(
        correlation_dimension(x, radii=radii), abs=2e-12
    )


def test_known_dimensions():
    x = np.linspace(0, 1, 2000)
    assert correlation_dimension(x, radii=np.geomspace(0.005, 0.05, 20)) == pytest.approx(1.0, abs=0.03)
    x = np.random.default_rng(2).normal(size=3000)
    assert correlation_dimension(x) == pytest.approx(2.0, abs=0.08)


@pytest.mark.parametrize("x", [0, "abc", [1j, 2j], [[1, 2], [3]], [], [1], [1, 2]])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        correlation_dimension(x)


@pytest.mark.parametrize("name", ["order", "delay"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, True, np.nan, "2"])
def test_invalid_integer(name, bad):
    with pytest.raises(ValueError, match=name):
        correlation_dimension(np.arange(20.0), **{name: bad})


@pytest.mark.parametrize("radii", [[], [1], [0, 1], [-1, 1], [2, 1], [1, 1], [[1, 2]], [1, np.inf], [np.nan, 1], [1j, 2j]])
def test_invalid_radii(radii):
    with pytest.raises(ValueError, match="radii"):
        correlation_dimension(np.arange(20.0), radii=radii)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_full_index(bad):
    x = np.zeros((2, 3, 20))
    x[1, 2, 7] = bad
    with pytest.raises(ValueError, match=r"index \(1, 2, 7\)"):
        correlation_dimension(x)
