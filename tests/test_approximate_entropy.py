"""Approximate entropy: literal Pincus formula, inclusive counts and batch parity."""

import numpy as np
import pytest

from complexeva.approximate_entropy import _tree_phi, approximate_entropy


def _literal(x, order=2, r=None):
    x = np.asarray(x, dtype=np.float64)
    radius = 0.2 * np.std(x) if r is None else r
    phi = []
    for dimension in (order, order + 1):
        windows = np.lib.stride_tricks.sliding_window_view(x, dimension)
        counts = (np.max(np.abs(windows[:, None] - windows[None, :]), axis=2) <= radius).sum(axis=1)
        phi.append(np.log(counts / len(windows)).mean())
    return phi[0] - phi[1]


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("r", [None, 0.0, 0.25, 2.0])
def test_literal_formula(order, r):
    x = np.random.default_rng(21).integers(-3, 4, size=67) / 4
    assert approximate_entropy(x, order=order, r=r) == pytest.approx(_literal(x, order, r), abs=2e-15)


def test_self_matches_final_template_and_closed_radius():
    x = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    assert approximate_entropy(x, r=1.0) == pytest.approx(_literal(x, r=1.0), abs=1e-15)
    assert approximate_entropy(x, r=0.0) == pytest.approx(_literal(x, r=0.0), abs=1e-15)
    assert approximate_entropy([0.0, 1.0], order=1, r=0.0) == -np.log(2)


@pytest.mark.parametrize("r", [None, 0.0, 1.0])
@pytest.mark.parametrize("n", [3, 100, 4100])
def test_constant_is_defined(n, r):
    assert approximate_entropy(np.zeros(n), r=r) == 0.0


@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
@pytest.mark.parametrize("r", [0.0, 0.25, 0.9, 5.0])
def test_tree_inclusive_counts(dimension, r):
    x = np.random.default_rng(13).integers(-4, 5, size=71) / 4
    windows = np.lib.stride_tricks.sliding_window_view(x, dimension)
    counts = (np.max(np.abs(windows[:, None] - windows[None, :]), axis=2) <= r).sum(axis=1)
    assert _tree_phi(x, dimension, r) == pytest.approx(np.log(counts / len(windows)).mean(), abs=2e-15)


@pytest.mark.parametrize("n", [37, 4200])
@pytest.mark.parametrize("r", [None, 0.3])
def test_batch_equals_each_1d_bitwise(n, r):
    rng = np.random.default_rng(48)
    base = rng.standard_normal((2, 3, 2 * n))
    layouts = [base[..., :n], base[..., ::2], np.asfortranarray(base[..., :n]), base[..., :n].astype(np.float32)]
    layouts += [np.rint(base[..., :n] * 3).astype(np.int16), base[..., :n] > 0]
    for x in layouts:
        actual = approximate_entropy(x, r=r)
        expected = np.array([approximate_entropy(x[index], r=r) for index in np.ndindex(x.shape[:-1])])
        assert isinstance(actual, np.ndarray)
        assert actual.shape == x.shape[:-1]
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual.ravel().view(np.uint64), expected.view(np.uint64))


def test_empty_batch_and_list():
    empty = approximate_entropy(np.empty((0, 3, 20)))
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (0, 3)
    assert isinstance(approximate_entropy(list(range(10))), float)


@pytest.mark.parametrize("bad", [True, 0, -1, 2.5, "2", None])
def test_invalid_order(bad):
    with pytest.raises(ValueError, match="order"):
        approximate_entropy(np.arange(10), order=bad)


@pytest.mark.parametrize("bad", [True, -0.1, np.inf, np.nan, "auto", 1j])
def test_invalid_radius(bad):
    with pytest.raises(ValueError, match="r"):
        approximate_entropy(np.arange(10), r=bad)


@pytest.mark.parametrize("x", [1.0, [], [1.0, 2.0], [1j, 2j, 3j], ["a", "b", "c"]])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        approximate_entropy(x)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_full_index(value):
    x = np.ones((2, 3, 10))
    x[1, 2, 4] = value
    with pytest.raises(ValueError, match=r"index \(1, 2, 4\)"):
        approximate_entropy(x)


@pytest.mark.parametrize("order", [2, 3])
def test_antropy_agreement(order):
    antropy = pytest.importorskip("antropy")
    x = np.random.default_rng(42).standard_normal(513)
    for r in (None, 0.0, 0.2, 2.0):
        assert approximate_entropy(x, order, r) == pytest.approx(antropy.app_entropy(x, order=order, tolerance=r), abs=5e-14)


def test_long_tree_antropy_agreement():
    antropy = pytest.importorskip("antropy")
    x = np.random.default_rng(29).standard_normal(5000)
    assert approximate_entropy(x) == pytest.approx(antropy.app_entropy(x), abs=5e-14)


def test_neurokit2_and_entropyhub_agreement():
    nk = pytest.importorskip("neurokit2")
    eh = pytest.importorskip("EntropyHub")
    x = np.random.default_rng(31).standard_normal(117)
    r = 0.2 * np.std(x, ddof=0)
    value = approximate_entropy(x, r=r)
    assert value == pytest.approx(nk.entropy_approximate(x, dimension=2, delay=1, tolerance=r)[0], abs=5e-14)
    assert value == pytest.approx(eh.ApEn(x, m=2, tau=1, r=r)[0][-1], abs=5e-14)
    # NeuroKit2 takes abs(Phi_m - Phi_m1), unlike the signed Pincus definition.
    signed = approximate_entropy(x, r=0.0)
    assert signed < 0
    assert abs(signed) == pytest.approx(nk.entropy_approximate(x, tolerance=0.0)[0], abs=5e-14)


def test_mne_features_sample_sd_convention():
    univariate = pytest.importorskip("mne_features.univariate")
    x = np.random.default_rng(12).standard_normal(251)
    # Its public API hard-codes sample SD; our default intentionally uses population SD.
    r = 0.2 * np.std(x, ddof=1)
    assert approximate_entropy(x, r=r) == pytest.approx(univariate.compute_app_entropy(x[None])[0], abs=5e-14)
