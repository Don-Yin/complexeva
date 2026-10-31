import inspect
from itertools import product

import numpy
import pytest

from complexeva.hjorth import hjorth


def _reference(x):
    """Unpadded population-variance definition used by antropy and NeuroKit2."""
    x = numpy.asarray(x, dtype=numpy.float64)
    first = numpy.diff(x)
    second = numpy.diff(first)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        mobility = numpy.sqrt(numpy.var(first) / numpy.var(x))
        complexity = numpy.sqrt(numpy.var(second) / numpy.var(first)) / mobility
    return mobility, complexity


def test_known_quadratic_values():
    mobility, complexity = hjorth([1, 2, 4, 7, 11])
    assert mobility == pytest.approx(numpy.sqrt(1.25 / 13.2), abs=1e-15)
    assert complexity == 0.0
    assert isinstance(mobility, float)
    assert isinstance(complexity, float)


@pytest.mark.parametrize("n", [3, 4, 7, 63, 64, 65, 66, 127, 128, 129, 1000, 10000, 100000])
@pytest.mark.parametrize("kind", ["noise", "walk", "sine", "binary", "offset", "trend"])
def test_matches_population_variance_formula(n, kind):
    rng = numpy.random.default_rng(72 + n)
    x = rng.normal(size=n)
    if kind == "walk":
        x = numpy.cumsum(x)
    elif kind == "sine":
        x = numpy.sin(numpy.arange(n) * 0.2)
    elif kind == "binary":
        x = numpy.resize([0.0, 1.0, 0.0, 0.0, 1.0], n)
    elif kind == "offset":
        x = 1e12 + x
    elif kind == "trend":
        x = numpy.arange(n) + x * 0.01
    # Center the high-offset reference before NumPy's reduction: centering is
    # mathematically invariant, while var(1e12 + noise) loses up to 1e-7.
    expected = _reference(x - x[0]) if kind == "offset" else _reference(x)
    numpy.testing.assert_allclose(hjorth(x), expected, rtol=2e-12, atol=2e-12, equal_nan=True)


def test_matches_antropy_and_neurokit2():
    antropy = pytest.importorskip("antropy")
    neurokit2 = pytest.importorskip("neurokit2")
    rng = numpy.random.default_rng(972)
    for n in (3, 7, 1000, 10000, 100000):
        for x in (rng.normal(size=n), numpy.cumsum(rng.normal(size=n))):
            actual = hjorth(x)
            numpy.testing.assert_allclose(actual, antropy.hjorth_params(x), rtol=1e-12, atol=1e-12)
            complexity, info = neurokit2.complexity_hjorth(x)
            numpy.testing.assert_allclose(actual, (info["Mobility"], complexity), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("value", [0.0, 4.0, -37.0, 1e100])
def test_constant_signal_is_undefined(value):
    numpy.testing.assert_array_equal(numpy.isnan(hjorth(numpy.full(40, value))), [True, True])


@pytest.mark.parametrize("step", [1.0, -2.0, 0.5])
def test_affine_signal_has_zero_mobility_and_undefined_complexity(step):
    mobility, complexity = hjorth(numpy.arange(40) * step + 3)
    assert mobility == 0.0
    assert numpy.isnan(complexity)


def test_white_noise_and_periodic_signal_limits():
    noise = numpy.random.default_rng(4).normal(size=100000)
    numpy.testing.assert_allclose(hjorth(noise), (numpy.sqrt(2), numpy.sqrt(1.5)), atol=0.01)
    sine = numpy.sin(numpy.arange(10000) * 2 * numpy.pi / 100)
    mobility, complexity = hjorth(sine)
    assert mobility == pytest.approx(2 * numpy.sin(numpy.pi / 100), abs=1e-5)
    assert complexity == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("length", range(3, 9))
def test_exhaustive_short_binary_signals(length):
    for x in product([0.0, 1.0], repeat=length):
        numpy.testing.assert_allclose(hjorth(x), _reference(x), atol=2e-15, rtol=2e-15, equal_nan=True)


@pytest.mark.parametrize("layout", ["C", "F", "reversed", "sliced", "windows"])
@pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32, numpy.int16, numpy.uint64, numpy.bool_, ">f8"])
def test_batch_rows_equal_scalar_bitwise(layout, dtype):
    rng = numpy.random.default_rng(92)
    x = rng.integers(0, 5, size=(2, 3, 101)).astype(dtype)
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    elif layout == "sliced":
        x = x[..., ::2]
    elif layout == "windows":
        x = numpy.lib.stride_tricks.sliding_window_view(rng.integers(0, 5, size=106).astype(dtype), 101)
    original = x.copy()
    x.flags.writeable = False
    actual = hjorth(x)
    expected = numpy.array([hjorth(x[index]) for index in numpy.ndindex(x.shape[:-1])])
    for i, result in enumerate(actual):
        assert isinstance(result, numpy.ndarray)
        assert result.shape == x.shape[:-1]
        assert result.dtype == numpy.float64
        expected_parameter = expected[:, i].reshape(x.shape[:-1]).copy()
        numpy.testing.assert_array_equal(result.view(numpy.uint64), expected_parameter.view(numpy.uint64))
    numpy.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("shape", [(0, 10), (2, 0, 10), (1, 10), (2, 3, 4, 10)])
def test_batch_shapes_including_empty_leading_axes(shape):
    for result in hjorth(numpy.ones(shape)):
        assert isinstance(result, numpy.ndarray)
        assert result.shape == shape[:-1]
        assert result.dtype == numpy.float64


def test_mixed_degenerate_batch_preserves_scalar_nan_bits():
    x = numpy.random.default_rng(7).normal(size=(4, 1000))
    x[1] = 1.0
    x[2] = numpy.arange(1000)
    x[3] = numpy.arange(1000) ** 2
    actual = hjorth(x)
    expected = numpy.asarray([hjorth(row) for row in x]).T.copy()
    for parameter, reference in zip(actual, expected, strict=True):
        assert isinstance(parameter, numpy.ndarray)
        numpy.testing.assert_array_equal(parameter.view(numpy.uint64), reference.view(numpy.uint64))


@pytest.mark.parametrize("bad", [numpy.nan, numpy.inf, -numpy.inf])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_nonfinite_error_names_full_first_index(bad, layout):
    x = numpy.ones((2, 3, 10))
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    x[1, 1, 7] = bad
    x[1, 2, 0] = numpy.nan
    with pytest.raises(ValueError, match=r"finite.*at index \(1, 1, 7\)"):
        hjorth(x)


@pytest.mark.parametrize("x", [[], [1], [1, 2], numpy.ones((3, 0)), numpy.ones((2, 3, 2))])
def test_short_time_axis_rejected(x):
    with pytest.raises(ValueError, match="at least 3 samples"):
        hjorth(x)


@pytest.mark.parametrize("x", [1.0, None, [1, 2, "bad"], [1, 2, 3j], [[1, 2], [3]]])
def test_nonreal_scalar_and_ragged_input_rejected(x):
    with pytest.raises(ValueError, match="x"):
        hjorth(x)


def test_signature_has_only_x():
    assert tuple(inspect.signature(hjorth).parameters) == ("x",)
