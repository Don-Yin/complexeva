import inspect
import warnings

import numpy as np
import pytest
from scipy.signal import periodogram, welch

from complexeva.spectral_entropy import spectral_entropy


def _reference(x, method="fft", nperseg=None, normalize=False):
    """The antropy definition, using the public SciPy estimators directly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        power = periodogram(x, 100)[1] if method == "fft" else welch(x, 100, nperseg=nperseg)[1]
    if not np.any(power):
        return np.nan
    probabilities = power / power.sum()
    positive = probabilities[probabilities > 0]
    entropy = -np.sum(positive * np.log2(positive))
    return entropy / np.log2(len(probabilities)) if normalize else entropy


@pytest.mark.parametrize("normalize", [False, True])
def test_known_one_sided_endpoint_weights(normalize):
    # The centred impulse has powers in the ratio 0:2:1 (DC:positive:Nyquist).
    expected = -(2 / 3 * np.log2(2 / 3) + 1 / 3 * np.log2(1 / 3))
    if normalize:
        expected /= np.log2(3)
    assert spectral_entropy([1, 0, 0, 0], 100, normalize=normalize) == pytest.approx(expected, abs=1e-15)


def test_uniform_non_dc_power_known_entropy():
    coefficients = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2), 1], dtype=complex)
    x = np.fft.irfft(coefficients, n=8)
    assert spectral_entropy(x, 100) == pytest.approx(2.0, abs=1e-14)
    assert spectral_entropy(x, 100, normalize=True) == pytest.approx(2 / np.log2(5), abs=1e-14)


@pytest.mark.parametrize("x", [[0, 1, 0, -1], [1, -1], [1, -1, 1, -1]])
def test_single_frequency_has_zero_entropy(x):
    assert spectral_entropy(x, 100) == 0.0


@pytest.mark.parametrize("n", [2, 3, 7, 31, 32, 255, 256, 257, 1000, 10000, 100000])
@pytest.mark.parametrize("method,nperseg", [("fft", None), ("welch", None), ("welch", 31), ("welch", 256)])
@pytest.mark.parametrize("normalize", [False, True])
def test_scipy_and_antropy_formula_agreement(n, method, nperseg, normalize):
    x = np.random.default_rng(36 + n).standard_normal(n)
    actual = spectral_entropy(x, 100, method, nperseg, normalize)
    assert actual == pytest.approx(_reference(x, method, nperseg, normalize), abs=1e-12)


@pytest.mark.parametrize("method", ["fft", "welch"])
@pytest.mark.parametrize("constant", [0.0, 0.1, 2.0, -1e300])
@pytest.mark.parametrize("normalize", [False, True])
def test_constant_is_undefined_even_when_mean_rounding_leaves_a_residual(method, constant, normalize):
    assert np.isnan(spectral_entropy(np.full(1000, constant), 100, method=method, normalize=normalize))


def test_welch_length_one_is_undefined_and_oversized_length_is_clipped():
    x = np.arange(17, dtype=float)
    assert np.isnan(spectral_entropy(x, 100, method="welch", nperseg=1, normalize=True))
    assert spectral_entropy(x, 100, method="welch", nperseg=100) == spectral_entropy(x, 100, method="welch", nperseg=17)
    # Like antropy, the periodogram ignores this Welch-only argument.
    assert spectral_entropy(x, 100, nperseg=0) == spectral_entropy(x, 100)


@pytest.mark.parametrize("method", ["fft", "welch"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "reversed", "stride", "windows"])
def test_batch_and_readonly_rows_match_scalar_bitwise(method, normalize, layout):
    rng = np.random.default_rng(27)
    x = rng.normal(size=(2, 3, 1001))
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    elif layout == "stride":
        x = x[..., ::2]
    elif layout == "windows":
        x = np.lib.stride_tricks.sliding_window_view(rng.normal(size=1006), 1001).reshape(2, 3, 1001)
    original = x.copy()
    x.flags.writeable = False
    actual = spectral_entropy(x, 100, method=method, normalize=normalize)
    expected = np.empty(x.shape[:-1])
    for index in np.ndindex(expected.shape):
        expected[index] = spectral_entropy(x[index], 100, method=method, normalize=normalize)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == x.shape[:-1]
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.uint64, np.float16, np.float32, np.float64, np.longdouble, ">f8"])
@pytest.mark.parametrize("method", ["fft", "welch"])
def test_batch_dtypes_share_float64_arithmetic(dtype, method):
    x = np.random.default_rng(23).integers(0, 5, size=(2, 3, 129)).astype(dtype)
    x[1, 1] = 2
    actual = spectral_entropy(x, 100, method=method)
    assert isinstance(actual, np.ndarray)
    expected = np.array([spectral_entropy(row, 100, method=method) for row in x.reshape(-1, 129)]).reshape(2, 3)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(spectral_entropy(x.tolist(), 100, method=method), actual)


@pytest.mark.parametrize("shape", [(0, 100), (2, 0, 100), (1, 100), (2, 3, 4, 33)])
@pytest.mark.parametrize("method", ["fft", "welch"])
def test_return_shape_including_empty_leading_dimensions(shape, method):
    x = np.random.default_rng(6).normal(size=shape)
    actual = spectral_entropy(x, 100, method=method)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == shape[:-1]
    assert actual.dtype == np.float64
    assert isinstance(spectral_entropy(np.arange(100), 100, method=method), float)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_nonfinite_input_names_the_first_full_index(invalid, layout):
    x = np.ones((2, 3, 40))
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    x[1, 1, 7] = invalid
    x[1, 2, 0] = np.nan
    with pytest.raises(ValueError, match=r"finite.*index \(1, 1, 7\)"):
        spectral_entropy(x, 100)


@pytest.mark.parametrize("x", [1, [], [1], np.ones((2, 0)), [1 + 1j, 2], ["a", "b"], [[1, 2], [3]]])
def test_invalid_signal_is_rejected(x):
    with pytest.raises(ValueError, match="x"):
        spectral_entropy(x, 100)


@pytest.mark.parametrize("sf", [0, -1, np.nan, np.inf, True, "100", None])
def test_invalid_sampling_frequency_is_rejected(sf):
    with pytest.raises(ValueError, match="sf"):
        spectral_entropy([0, 1], sf)


@pytest.mark.parametrize("length", [0, -1, 2.5, True, np.nan])
def test_invalid_welch_length_is_rejected(length):
    with pytest.raises(ValueError, match="nperseg"):
        spectral_entropy([0, 1], 100, method="welch", nperseg=length)


@pytest.mark.parametrize("method", ["periodogram", "FFT", "invalid", None, 0])
def test_invalid_method_is_rejected(method):
    with pytest.raises(ValueError, match="method"):
        spectral_entropy([0, 1], 100, method=method)


@pytest.mark.parametrize("method", ["fft", "welch"])
def test_frequency_and_extreme_amplitude_scalings_cancel(method):
    x = np.random.default_rng(77).normal(size=1000)
    expected = spectral_entropy(x, 100, method)
    for scale in [1e-300, 1e-150, -2.0, 1e150, 1e300]:
        assert spectral_entropy(x * scale, 250, method) == pytest.approx(expected, abs=1e-12)
    assert spectral_entropy(x, np.float64(100), method, np.int64(200)) == pytest.approx(
        spectral_entropy(x, 100, method, 200), abs=0
    )


def test_public_signal_parameter_and_no_axis_argument():
    signature = inspect.signature(spectral_entropy)
    assert next(iter(signature.parameters)) == "x"
    assert "axis" not in signature.parameters
