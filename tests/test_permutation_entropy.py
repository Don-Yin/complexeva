import importlib
import inspect
import re
from itertools import product
from math import factorial, log2

import numpy
import pytest

from complexeva import permutation_entropy


def test_known_patterns_match_antropy():
    # antropy 0.2.3: five order-3 patterns have probabilities 2/5, 2/5, 1/5.
    x = [4, 7, 9, 10, 6, 11, 3]
    assert permutation_entropy(x) == pytest.approx(1.5219280948873621, abs=1e-14)
    assert permutation_entropy(x, normalize=True) == pytest.approx(0.588762155916294, abs=1e-14)
    assert permutation_entropy(x, order=2) == pytest.approx(0.9182958340544896, abs=1e-14)


def test_delay_changes_embedding_not_starting_step():
    assert permutation_entropy([4, 7, 9, 10, 6, 11, 3], delay=2) == pytest.approx(numpy.log2(3), abs=1e-14)


def test_ties_retain_temporal_order():
    # Stable sorting gives five different ordinal patterns for these windows.
    assert permutation_entropy([1, 1, 0, 1, 0, 0, 1]) == pytest.approx(numpy.log2(5), abs=1e-14)


@pytest.mark.parametrize("x", [numpy.arange(30), -numpy.arange(30), numpy.ones(30)])
@pytest.mark.parametrize("normalize", [False, True])
def test_monotonic_and_constant_inputs_have_zero_entropy(x, normalize):
    assert permutation_entropy(x, normalize=normalize) == 0.0


def test_uniform_patterns_normalize_to_one():
    assert permutation_entropy([0, 1, 0], order=2, normalize=True) == 1.0


def test_single_delayed_pattern_is_valid():
    assert permutation_entropy([4, 1, 7, 2, 0], order=3, delay=2) == 0.0


def test_large_order_does_not_overflow_an_integer_pattern_hash():
    x = numpy.random.default_rng(4).standard_normal(100)
    assert permutation_entropy(x, order=20) == pytest.approx(numpy.log2(81), abs=1e-14)


def test_white_noise_normalized_entropy_is_near_one():
    x = numpy.random.default_rng(0).standard_normal(8192)
    assert permutation_entropy(x, order=5, normalize=True) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("order", [1, 0, -1, 2.5, True, numpy.nan])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="order"):
        permutation_entropy(numpy.ones(20), order=order)


@pytest.mark.parametrize("delay", [0, -1, 1.5, True, numpy.nan])
def test_invalid_delay_rejected(delay):
    with pytest.raises(ValueError, match="delay"):
        permutation_entropy(numpy.ones(20), delay=delay)


@pytest.mark.parametrize("x, order, delay", [([], 3, 1), ([1, 2], 3, 1), ([1, 2, 3, 4], 3, 2)])
def test_short_series_rejected(x, order, delay):
    with pytest.raises(ValueError, match="samples"):
        permutation_entropy(x, order=order, delay=delay)


def test_numpy_integer_parameters_are_accepted():
    assert permutation_entropy(numpy.ones(20), order=numpy.int64(3), delay=numpy.int64(2)) == 0.0


def _reference(x, order, delay, normalize):
    """Evaluate the pre-optimization formula without an integer encoding."""
    values = numpy.asarray(x, dtype=numpy.float64)
    window = (order - 1) * delay + 1
    embedded = numpy.lib.stride_tricks.sliding_window_view(values, window)[:, ::delay]
    patterns = numpy.argsort(embedded, axis=1, kind="stable")
    _, counts = numpy.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    result = float(-numpy.sum(p * numpy.log2(p)))
    return result / log2(factorial(order)) if normalize else result


@pytest.mark.parametrize("n", [1000, 10000, 100000])
@pytest.mark.parametrize("order", range(3, 8))
@pytest.mark.parametrize("delay", [1, 2, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_permutation_entropy_requested_grid(n, order, delay, normalize):
    x = numpy.random.default_rng(20261029 + n).normal(size=n)
    assert abs(permutation_entropy(x, order, delay, normalize) - _reference(x, order, delay, normalize)) < 1e-12


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7, 8, 12, 20])
@pytest.mark.parametrize("delay", [1, 2, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_permutation_entropy_ties_dtypes_and_fallback(order, delay, normalize):
    rng = numpy.random.default_rng(654)
    x = rng.normal(size=257)
    cases = [
        rng.integers(-2, 3, size=257),
        numpy.ones(257),
        numpy.arange(257),
        -numpy.arange(257),
        numpy.round(x),
        x.astype(numpy.float32),
        x > 0,
        x.tolist(),
        x[::-1],
        x[::2],
        numpy.resize([0.0, -0.0, 1.0, -1.0], 257),
        1e15 + numpy.round(x),
        x * 1e-200,
        numpy.resize([1e308, -1e308, 0.0], 257),
    ]
    for case in cases:
        expected = _reference(case, order, delay, normalize)
        assert abs(permutation_entropy(case, order, delay, normalize) - expected) < 1e-12


@pytest.mark.parametrize("order", range(2, 8))
def test_permutation_entropy_exhaustive_binary_ties(order):
    for values in product([0.0, 1.0], repeat=9):
        for normalize in (False, True):
            expected = _reference(values, order, 1, normalize)
            assert abs(permutation_entropy(values, order, 1, normalize) - expected) < 1e-12


@pytest.mark.parametrize("order", range(2, 9))
@pytest.mark.parametrize("delay", [1, 2, 3])
def test_permutation_entropy_one_window(order, delay):
    x = numpy.random.default_rng(245).integers(-2, 3, size=(order - 1) * delay + 1)
    assert permutation_entropy(x, order, delay) == 0.0
    assert permutation_entropy(x, order, delay, normalize=True) == 0.0


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7, 8, 12, 20])
@pytest.mark.parametrize("delay", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", ["noise", "ties", "constant"])
def test_batch_matches_scalar_bitwise(order, delay, normalize, kind):
    rng = numpy.random.default_rng(21)
    x = rng.normal(size=(2, 3, 257))
    if kind == "ties":
        x = numpy.round(x)
    elif kind == "constant":
        x.fill(4.0)
    for values in (x, numpy.asfortranarray(x), x[..., ::-1], x[..., ::2]):
        actual = permutation_entropy(values, order, delay, normalize)
        assert isinstance(actual, numpy.ndarray)
        expected = numpy.array(
            [permutation_entropy(row, order, delay, normalize) for row in values.reshape(-1, values.shape[-1])]
        ).reshape(values.shape[:-1])
        assert actual.shape == values.shape[:-1]
        assert actual.dtype == numpy.float64
        numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("samples", [1000, 100000])
@pytest.mark.parametrize("layout", ["C", "F", "windows", "reversed"])
def test_batch_large_strided_rows_preserve_bits(monkeypatch, samples, layout):
    module = importlib.import_module("complexeva.permutation_entropy")
    rng = numpy.random.default_rng(81)
    x = rng.normal(size=(8, samples))
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "windows":
        x = numpy.lib.stride_tricks.sliding_window_view(rng.normal(size=samples + 7), samples)
    elif layout == "reversed":
        x = x[:, ::-1]
    x.flags.writeable = False
    kernel = module._PERMUTATION_ROWS[3]
    calls = []

    def checked(*args):
        calls.append(True)
        assert (args[1] is None) == (layout in ("F", "reversed"))
        return kernel(*args)

    monkeypatch.setitem(module._PERMUTATION_ROWS, 3, checked)
    actual = numpy.asarray(permutation_entropy(x, normalize=True))
    expected = numpy.array([permutation_entropy(row, normalize=True) for row in x])
    assert calls
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("order", [3, 8])
def test_batch_invalid_row_after_zero_entropy_row(order):
    x = numpy.ones((3, 100))[:, ::-1]
    x[1, 17] = numpy.inf
    x[2, 0] = numpy.nan
    with pytest.raises(ValueError, match=r"finite.*index \(1, 17\)"):
        permutation_entropy(x, order)


@pytest.mark.parametrize("shape", [(0, 100), (1, 100), (2, 0, 100), (2, 3, 4, 100)])
@pytest.mark.parametrize("order", [3, 8])
def test_batch_result_shape_including_empty_leading_axes(shape, order):
    x = numpy.random.default_rng(1).normal(size=shape)
    actual = permutation_entropy(x, order)
    assert isinstance(actual, numpy.ndarray)
    assert actual.shape == shape[:-1]
    assert actual.dtype == numpy.float64
    assert isinstance(permutation_entropy(numpy.arange(100), order), float)


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.int16, numpy.bool_, ">f8"])
def test_batch_numeric_dtypes_and_readonly_inputs(dtype):
    x = numpy.random.default_rng(4).integers(-2, 3, size=(3, 100)).astype(dtype)
    original = x.copy()
    x.flags.writeable = False
    actual = permutation_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([permutation_entropy(row) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))
    numpy.testing.assert_array_equal(x, original)
    numpy.testing.assert_array_equal(permutation_entropy(x.tolist()), actual)


def test_batch_constant_and_monotonic_rows_follow_scalar_rule():
    x = numpy.random.default_rng(5).normal(size=(4, 100))
    x[1] = 4.0
    x[2] = numpy.arange(100)
    x[3] = -numpy.arange(100)
    actual = permutation_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    assert actual[0] == permutation_entropy(x[0])
    numpy.testing.assert_array_equal(actual[1:].view(numpy.uint64), numpy.full(3, -0.0).view(numpy.uint64))


@pytest.mark.parametrize("invalid", [numpy.nan, numpy.inf, -numpy.inf])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_batch_nonfinite_error_names_first_full_index(invalid, layout):
    x = numpy.zeros((2, 3, 40))
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    x[1, 1, 7] = invalid
    x[1, 2, 0] = numpy.nan
    with pytest.raises(ValueError) as error:
        permutation_entropy(x)
    assert "finite" in str(error.value)
    assert repr(x[1, 1, 7]) in str(error.value)
    assert "at index (1, 1, 7)" in str(error.value)


@pytest.mark.parametrize("shape", [(100, 2), (2, 3, 4), (0, 0)])
def test_batch_length_validation_uses_last_axis(shape):
    with pytest.raises(ValueError, match=f"samples, got {shape[-1]}"):
        permutation_entropy(numpy.ones(shape), order=3, delay=2)


def test_scalar_input_names_bad_shape_and_no_axis_argument():
    with pytest.raises(ValueError, match=re.escape("shape ()")):
        permutation_entropy(1.0)
    assert "axis" not in inspect.signature(permutation_entropy).parameters
