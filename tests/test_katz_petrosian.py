"""Known values, package agreement, and last-axis invariants for both fractal dimensions."""

import inspect
from itertools import product

import numpy as np
import pytest

from complexeva.katz_petrosian import katz_fd, petrosian_fd

FUNCTIONS = (katz_fd, petrosian_fd)


def _katz_reference(x):
    """Evaluate Katz's planar Euclidean definition independently with NumPy arrays."""
    x = np.asarray(x, dtype=np.float64)
    length = np.hypot(1.0, np.diff(x)).sum()
    diameter = np.hypot(np.arange(len(x)), x - x[0]).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_steps = np.log10(len(x) - 1)
        return log_steps / (log_steps + np.log10(diameter / length))


def _petrosian_reference(x):
    """Evaluate derivative sign-bit changes and use the original input length."""
    x = np.asarray(x, dtype=np.float64)
    symbols = np.signbit(np.diff(x))
    changes = np.count_nonzero(symbols[1:] != symbols[:-1])
    n = len(x)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * changes)))


def test_known_planar_path_and_derivative_signs():
    # Katz: L = 2*sqrt(2) + sqrt(5) + sqrt(10), d = sqrt(17). Petrosian: three inversions.
    x = [0, 1, -1, 2, 1]
    length = 2 * np.sqrt(2) + np.sqrt(5) + np.sqrt(10)
    expected = np.log10(4) / (np.log10(4) + np.log10(np.sqrt(17) / length))
    assert katz_fd(x) == pytest.approx(expected, abs=1e-14)
    assert petrosian_fd(x) == pytest.approx(np.log10(5) / (np.log10(5) + np.log10(5 / 6.2)), abs=1e-14)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("n", [3, 17, 1000, 10000, 100000])
def test_monotonic_signal_has_dimension_one(function, n):
    assert function(np.arange(n)) == pytest.approx(1.0, abs=1e-12)
    assert function(-np.arange(n)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", range(2, 10))
def test_planar_short_signals_and_unrolled_tail(n):
    x = np.random.default_rng(n).normal(size=(3, n))
    actual = katz_fd(x)
    assert isinstance(actual, np.ndarray)
    expected = np.array([katz_fd(row) for row in x])
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    for row, value in zip(x, actual):
        np.testing.assert_allclose(value, _katz_reference(row), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_planar_time_coordinates_make_amplitude_scaling_meaningful():
    x = np.array([0.0, 1.0, -1.0, 2.0, 1.0])
    assert abs(katz_fd(x) - katz_fd(10 * x)) > 0.1
    assert katz_fd(x + 50) == katz_fd(x)


def test_constant_and_shortest_defined_inputs():
    assert katz_fd(np.ones(20)) == 1.0
    assert np.isnan(katz_fd([2.0, 3.0]))
    assert katz_fd([0.0, 1.0, 0.0]) == pytest.approx(2.0, abs=1e-14)
    assert petrosian_fd(np.ones(20)) == 1.0
    assert petrosian_fd([2.0, 3.0]) == 1.0


@pytest.mark.parametrize("function,reference", [(katz_fd, _katz_reference), (petrosian_fd, _petrosian_reference)])
@pytest.mark.parametrize("n", [17, 1000, 10000, 100000])
def test_numpy_reference_on_noise_walk_and_ties(function, reference, n):
    rng = np.random.default_rng(n)
    noise = rng.normal(size=n)
    for x in (noise, noise.cumsum(), np.round(noise), noise[::-1], noise[::2]):
        assert function(x) == pytest.approx(reference(x), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("function,reference", [(katz_fd, _katz_reference), (petrosian_fd, _petrosian_reference)])
def test_exhaustive_binary_sequences_and_signed_zero(function, reference):
    for x in product((0.0, 1.0), repeat=7):
        np.testing.assert_allclose(function(x), reference(x), rtol=1e-12, atol=1e-12, equal_nan=True)
    for x in ([1.0, 0.0, 0.0, 1.0], [0.0, -0.0, 0.0, -0.0, 1.0], [0.0, -1.0, -1.0, -2.0]):
        np.testing.assert_allclose(function(x), reference(x), rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "slice", "transpose", "windows"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.uint64, np.bool_, np.float16, ">f8"])
def test_batch_matches_scalar_bitwise_for_every_dtype_and_layout(function, layout, dtype):
    rng = np.random.default_rng(34)
    x = rng.integers(0, 7, size=(2, 3, 37)).astype(dtype)
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[..., ::-1]
    elif layout == "slice":
        x = x[..., ::2]
    elif layout == "transpose":
        x = x.swapaxes(0, 1)
    elif layout == "windows":
        x = np.lib.stride_tricks.sliding_window_view(x[0, 0], 32)
    original = x.copy()
    x.flags.writeable = False
    actual = function(x)
    expected = np.array([function(x[index]) for index in np.ndindex(x.shape[:-1])]).reshape(x.shape[:-1])
    assert actual.dtype == np.float64
    assert actual.shape == x.shape[:-1]
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_integer_samples_are_cast_before_subtraction(function):
    x = np.array([[0, 2**64 - 1, 1, 2**63, 0]], dtype=np.uint64)
    actual = function(x)
    expected = np.array([function(x[0])])
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("shape", [(0, 20), (2, 0, 20), (1, 20), (2, 3, 4, 20)])
def test_result_shape_including_empty_leading_axes(function, shape):
    result = function(np.ones(shape))
    assert result.shape == shape[:-1]
    assert result.dtype == np.float64
    assert isinstance(function(np.arange(20)), float)
    assert list(inspect.signature(function).parameters) == ["x"]


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
def test_first_nonfinite_sample_names_full_index(function, invalid, layout):
    x = np.ones((2, 3, 40))
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[..., ::-1]
    x[1, 1, 7] = invalid
    x[1, 2, 0] = np.nan
    with pytest.raises(ValueError, match=r"finite.*at index \(1, 1, 7\)"):
        function(x)
    with pytest.raises(ValueError, match="finite.*at index 7"):
        function(x[1, 1])


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("x", [1.0, [], [1], np.empty((2, 0)), np.ones((2, 1)), [1j, 2j], ["a", "b"], [[1, 2], [3]]])
def test_invalid_input_is_value_error(function, x):
    with pytest.raises(ValueError, match="x"):
        function(x)


def test_antropy_agreement():
    antropy = pytest.importorskip("antropy")
    rng = np.random.default_rng(34)
    for n in (17, 1000, 10000, 100000):
        x = rng.normal(size=n)
        assert petrosian_fd(x) == pytest.approx(antropy.petrosian_fd(x), rel=1e-12, abs=1e-12)
        # antropy drops time coordinates and therefore is not a Katz-1988 reference.
        assert abs(katz_fd(x) - antropy.katz_fd(x)) > 1.0


def test_neurokit2_and_mne_features_agreement():
    neurokit2 = pytest.importorskip("neurokit2")
    mne_features = pytest.importorskip("mne_features.univariate")
    x = np.random.default_rng(34).normal(size=1000)
    assert abs(katz_fd(x) - neurokit2.fractal_katz(x)[0]) > 1.0
    assert abs(katz_fd(x) - mne_features.compute_katz_fd(x[None, :])[0]) > 1.0
    assert katz_fd(x) == pytest.approx(_katz_reference(x), rel=1e-12, abs=1e-12)
    symbols = np.signbit(np.diff(x))
    # Repeating the last symbol restores N symbols without adding an inversion.
    matched = np.r_[symbols, symbols[-1]]
    assert petrosian_fd(x) == neurokit2.fractal_petrosian(matched, symbolize=None)[0]
    assert abs(petrosian_fd(x) - neurokit2.fractal_petrosian(x)[0]) > 1e-8
