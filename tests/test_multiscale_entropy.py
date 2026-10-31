import numpy as np
import pytest

from complexeva.multiscale_entropy import multiscale_entropy
from complexeva.sample_entropy import sample_entropy


def _literal_entropy(x, scale, order, r):
    """Independent finite-pair definition, with incomplete coarse-graining blocks dropped."""
    y = np.array([np.mean(x[j : j + scale]) for j in range(0, len(x) - scale + 1, scale)])
    count_m = count_m1 = 0
    for i in range(len(y) - order):
        for j in range(i + 1, len(y) - order):
            if max(abs(y[i : i + order] - y[j : j + order])) < r:
                count_m += 1
                count_m1 += abs(y[i + order] - y[j + order]) < r
    return np.nan if count_m == 0 else np.inf if count_m1 == 0 else -np.log(count_m1 / count_m)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_literal_definition(order):
    x = np.random.default_rng(71).standard_normal(89)
    scales = [1, 2, 5, 2]
    r = 0.7
    expected = [_literal_entropy(x, scale, order, r) for scale in scales]
    np.testing.assert_allclose(multiscale_entropy(x, scales, order, r), expected, rtol=1e-14, atol=1e-14)


def test_default_parameters_from_costa_2002():
    x = np.random.default_rng(91).normal(size=1000)
    result = multiscale_entropy(x)
    assert result.shape == (20,)
    np.testing.assert_array_equal(result, multiscale_entropy(x, range(1, 21), order=2, r=0.15 * np.std(x)))


def test_threshold_is_fixed_from_original_signal():
    x = np.random.default_rng(17).normal(size=3000)
    result = multiscale_entropy(x, scales=[1, 3, 8])
    expected = []
    adaptive = []
    for scale in [1, 3, 8]:
        coarse = x[: (len(x) // scale) * scale].reshape(-1, scale).mean(axis=1)
        expected.append(sample_entropy(coarse, order=2, tolerance=0.15 * np.std(x)))
        adaptive.append(sample_entropy(coarse, order=2, tolerance=0.15 * np.std(coarse)))
    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)
    assert abs(result[-1] - adaptive[-1]) > 0.2


@pytest.mark.parametrize("r", [None, 0, 0.7])
@pytest.mark.parametrize("layout", ["contiguous", "windows", "reverse", "float32", "integer", "fortran", "higher-strided"])
def test_batch_bitwise_matches_scalar_loop(layout, r):
    base = np.random.default_rng(5).normal(size=(2, 3, 160))
    x = base
    if layout == "windows":
        x = np.lib.stride_tricks.sliding_window_view(base[0, 0], 80)[::7]
    elif layout == "reverse":
        x = base[:, :, ::-1]
    elif layout == "float32":
        x = base.astype(np.float32)
    elif layout == "integer":
        x = (10 * base).astype(np.int16)
    elif layout == "fortran":
        x = np.asfortranarray(base)
    elif layout == "higher-strided":
        x = base.transpose(1, 0, 2)[:, :, ::2]
    scales = [1, 2, 5, 2]
    result = multiscale_entropy(x, scales, r=r)
    expected = np.array([multiscale_entropy(x[index], scales, r=r) for index in np.ndindex(x.shape[:-1])])
    expected = expected.reshape(x.shape[:-1] + (len(scales),))
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


def test_long_batch_uses_same_tree_counts():
    x = np.random.default_rng(8).normal(size=(2, 10000))
    result = multiscale_entropy(x, scales=[1, 2, 3, 7])
    expected = np.array([multiscale_entropy(row, scales=[1, 2, 3, 7]) for row in x])
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


def test_coarse_graining_drops_incomplete_final_block():
    x = np.random.default_rng(31).normal(size=99)
    result = multiscale_entropy(x, scales=[4], r=0.4)
    assert result[0] == multiscale_entropy(x[:96], scales=[4], r=0.4)[0]


def test_constant_and_zero_tolerance():
    assert np.isnan(multiscale_entropy(np.ones(100))).all()
    assert np.isnan(multiscale_entropy(np.arange(100), r=0)).all()
    np.testing.assert_array_equal(multiscale_entropy(np.ones(100), r=0.1), np.zeros(20))


@pytest.mark.parametrize("value", [0.0, 0.1, -3.1, 1e300])
def test_exact_constants_have_undefined_default_entropy(value):
    x = np.full((2, 100), value)
    assert np.isnan(multiscale_entropy(x[0])).all()
    assert np.isnan(multiscale_entropy(x)).all()


def test_strict_threshold_excludes_boundary_matches():
    assert np.isnan(multiscale_entropy(np.arange(100), scales=2, r=1)).all()


def test_no_matches_and_no_extensions():
    assert np.isnan(multiscale_entropy(np.arange(50), scales=1, r=0.01)[0])
    x = np.array([0, 0, 2, 0, 0, 5, 8, 10], dtype=float)
    assert np.isinf(multiscale_entropy(x, scales=1, r=0.1)[0])


def test_empty_batch():
    assert multiscale_entropy(np.empty((0, 100))).shape == (0, 20)
    assert multiscale_entropy(np.empty((2, 0, 100)), scales=[1, 5]).shape == (2, 0, 2)


@pytest.mark.parametrize("x", [0.1, [], [1, 2, 3], [1 + 2j] * 100, ["bad"] * 100, [[1], [2, 3]]])
def test_invalid_input(x):
    with pytest.raises(ValueError):
        multiscale_entropy(x, scales=1)


@pytest.mark.parametrize("scales", [0, -1, True, 1.0, [], [[1, 2]], [1.0, 2.0], [True], [1, 0], [1, 100]])
def test_invalid_scales(scales):
    with pytest.raises(ValueError, match="scale"):
        multiscale_entropy(np.ones(100), scales=scales)


@pytest.mark.parametrize("order", [0, -1, True, 2.0, np.nan])
def test_invalid_order(order):
    with pytest.raises(ValueError, match="order"):
        multiscale_entropy(np.ones(100), scales=1, order=order)


@pytest.mark.parametrize("r", [-0.1, np.inf, np.nan, True, "bad", 1j])
def test_invalid_tolerance(r):
    with pytest.raises(ValueError, match="r"):
        multiscale_entropy(np.ones(100), r=r)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_full_index(value):
    x = np.zeros((2, 3, 100))
    x[1, 2, 17] = value
    with pytest.raises(ValueError, match=r"index \(1, 2, 17\)"):
        multiscale_entropy(x)
    with pytest.raises(ValueError, match=r"index \(2, 1, 17\)"):
        multiscale_entropy(x.transpose(1, 0, 2))


def test_pyentrp_agreement():
    entropy = pytest.importorskip("pyentrp.entropy")
    x = np.random.default_rng(19).normal(size=1000)
    r = 0.15 * np.std(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = entropy.multiscale_entropy(x, sample_length=3, tolerance=r, maxscale=10)
    np.testing.assert_allclose(multiscale_entropy(x, 10, r=r), expected, atol=1e-14, rtol=1e-14)


def test_neurokit2_agreement():
    nk = pytest.importorskip("neurokit2")
    x = np.random.default_rng(20).normal(size=1000)
    r = 0.15 * np.std(x)
    _, info = nk.entropy_multiscale(x, scale=list(range(1, 11)), dimension=2, tolerance=r, method="MSEn")
    np.testing.assert_allclose(multiscale_entropy(x, 10, r=r), info["Value"], atol=1e-14, rtol=1e-14)


def test_entropyhub_agreement(capsys):
    eh = pytest.importorskip("EntropyHub")
    x = np.random.default_rng(21).normal(size=800)
    r = 0.15 * np.std(x)
    expected, _ = eh.MSEn(x, eh.MSobject("SampEn", m=2, r=r), Scales=8, Methodx="coarse", RadNew=0)
    capsys.readouterr()
    np.testing.assert_allclose(multiscale_entropy(x, 8, r=r), expected, atol=1e-14, rtol=1e-14)
