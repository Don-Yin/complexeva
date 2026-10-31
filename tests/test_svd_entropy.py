import inspect

import numpy
import pytest

from complexeva.svd_entropy import svd_entropy


def _reference(x, order=3, delay=1, normalize=False):
    """Evaluate Roberts et al.'s singular-value definition directly."""
    x = numpy.asarray(x, dtype=numpy.float64)
    embedded = numpy.lib.stride_tricks.sliding_window_view(x, (order - 1) * delay + 1)[:, ::delay]
    singular = numpy.linalg.svd(embedded, compute_uv=False)
    total = singular.sum()
    if total == 0.0:
        return numpy.nan
    probabilities = singular[singular > 0.0] / total
    entropy = -numpy.sum(probabilities * numpy.log2(probabilities))
    return entropy / numpy.log2(order) if normalize else entropy


def test_known_antropy_example():
    x = [4, 7, 9, 10, 6, 11, 3]
    assert svd_entropy(x, order=2) == pytest.approx(0.7618909465130066, abs=1e-14, rel=0)
    assert svd_entropy(x, normalize=True) == pytest.approx(0.6870083043946692, abs=1e-14, rel=0)


@pytest.mark.parametrize("order", [2, 3, 5, 10, 20])
@pytest.mark.parametrize("delay", [1, 2, 5])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", ["noise", "binary", "sine", "line", "nearconstant", "correlated"])
def test_formula_for_full_rank_and_ill_conditioned_signals(order, delay, normalize, kind):
    x = numpy.random.default_rng(37).normal(size=301)
    if kind == "binary":
        x = numpy.round(x)
    elif kind == "sine":
        x = numpy.sin(numpy.arange(x.size) * 0.137)
    elif kind == "line":
        x = numpy.arange(x.size, dtype=numpy.float64)
    elif kind == "nearconstant":
        x = 1 + 1e-7 * x
    elif kind == "correlated":
        x = numpy.cumsum(x)
    assert svd_entropy(x, order, delay, normalize) == pytest.approx(_reference(x, order, delay, normalize), abs=1e-12, rel=0)


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (5, 3), (10, 1)])
def test_requested_lengths(n, order, delay):
    x = numpy.random.default_rng(n).normal(size=n)
    assert svd_entropy(x, order, delay) == pytest.approx(_reference(x, order, delay), abs=1e-12, rel=0)


@pytest.mark.parametrize("kind", ["sine", "line", "nearconstant"])
def test_large_ill_conditioned_embeddings_preserve_small_singular_values(kind):
    x = numpy.arange(100_000, dtype=numpy.float64)
    if kind == "sine":
        x = numpy.sin(x * 0.1)
    elif kind == "nearconstant":
        x = 1 + numpy.random.default_rng(9).normal(size=x.size) * 1e-7
    assert svd_entropy(x) == pytest.approx(_reference(x), abs=1e-12, rel=0)


@pytest.mark.parametrize("constant", [-1e300, -2.0, 1e-300, 1.0])
def test_nonzero_constant_has_defined_zero_entropy(constant):
    assert svd_entropy(numpy.full(30, constant)) == 0.0


def test_all_zero_embedding_is_undefined():
    assert numpy.isnan(svd_entropy(numpy.zeros(30)))
    assert numpy.isnan(svd_entropy([0.0, 1.0, 0.0, 2.0, 0.0], order=3, delay=2))


def test_unused_samples_do_not_set_the_embedding_scale():
    x = numpy.array([1e-300, 1e300, 2e-300, 1e300, 3e-300])
    assert svd_entropy(x, order=3, delay=2) == 0.0
    x = numpy.random.default_rng(7).normal(size=14) * 1e-300
    x[[2, 3, 6, 7, 10, 11]] = 1e300
    embedded = numpy.array([[x[start + j * 4] / 1e-300 for j in range(4)] for start in range(2)])
    singular = numpy.linalg.svd(embedded, compute_uv=False)
    probability = singular / singular.sum()
    expected = -numpy.sum(probability * numpy.log2(probability))
    assert svd_entropy(x, order=4, delay=4) == pytest.approx(expected, abs=1e-12, rel=0)


@pytest.mark.parametrize("order,delay", [(3, 1), (5, 3), (20, 2)])
def test_one_embedding_row(order, delay):
    x = numpy.random.default_rng(9).normal(size=(order - 1) * delay + 1)
    assert svd_entropy(x, order, delay) == pytest.approx(0.0, abs=1e-14, rel=0)


@pytest.mark.parametrize("order,delay,count", [(3, 1, 2), (20, 1, 4), (20, 3, 4), (1_000, 2, 3), (10_000, 1, 2)])
def test_wide_embeddings_use_the_same_definition(order, delay, count):
    x = numpy.random.default_rng(18).normal(size=(order - 1) * delay + count)
    assert svd_entropy(x, order, delay) == pytest.approx(_reference(x, order, delay), abs=1e-12, rel=0)


@pytest.mark.parametrize("factor", [1e-300, 1e-150, 1e150, 1e300, -1e-300, -1e300])
def test_extreme_rescaling_does_not_overflow_or_underflow(factor):
    x = numpy.random.default_rng(9).normal(size=301)
    assert svd_entropy(factor * x) == pytest.approx(_reference(x), abs=1e-12, rel=0)
    x = numpy.sin(numpy.arange(x.size) * 0.137)
    assert svd_entropy(factor * x) == pytest.approx(_reference(x), abs=1e-12, rel=0)


@pytest.mark.parametrize("order", [1, 0, -1, 2.5, True, numpy.nan])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="order"):
        svd_entropy(numpy.ones(20), order=order)


@pytest.mark.parametrize("delay", [0, -1, 1.5, True, numpy.nan])
def test_invalid_delay_rejected(delay):
    with pytest.raises(ValueError, match="delay"):
        svd_entropy(numpy.ones(20), delay=delay)


@pytest.mark.parametrize("x,order,delay", [([], 3, 1), ([1, 2], 3, 1), ([1, 2, 3, 4], 3, 2)])
def test_short_series_rejected(x, order, delay):
    with pytest.raises(ValueError, match="samples"):
        svd_entropy(x, order=order, delay=delay)


@pytest.mark.parametrize("x", [1.0, ["a", "b", "c"], [1j, 0j, 2j], [[1, 2], [3]]])
def test_invalid_signal_rejected(x):
    with pytest.raises(ValueError):
        svd_entropy(x)


def test_numpy_integer_parameters_and_signature():
    assert svd_entropy(numpy.ones(20), numpy.int64(3), numpy.int64(2)) == 0.0
    assert list(inspect.signature(svd_entropy).parameters) == ["x", "order", "delay", "normalize"]


@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (5, 3), (20, 2)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "reversed", "sliced", "transposed"])
def test_batch_rows_equal_scalar_bitwise(order, delay, normalize, layout):
    x = numpy.random.default_rng(21).normal(size=(2, 3, 103))
    x[0, 1] = numpy.sin(numpy.arange(103) * 0.1)
    x[1, 0] = 0.0
    x[1, 1] = 4.0
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    elif layout == "sliced":
        x = x[..., ::2]
    elif layout == "transposed":
        x = x.swapaxes(0, 1)
    x.flags.writeable = False
    actual = svd_entropy(x, order, delay, normalize)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.empty(x.shape[:-1])
    for index in numpy.ndindex(expected.shape):
        expected[index] = svd_entropy(x[index], order, delay, normalize)
    assert actual.shape == x.shape[:-1]
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.int16, numpy.bool_, ">f8"])
def test_batch_dtypes_and_readonly_inputs(dtype):
    x = numpy.random.default_rng(4).integers(-2, 3, size=(3, 100)).astype(dtype)
    original = x.copy()
    x.flags.writeable = False
    actual = svd_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([svd_entropy(row) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))
    numpy.testing.assert_array_equal(x, original)
    numpy.testing.assert_array_equal(svd_entropy(x.tolist()), actual)


def test_overlapping_windows_preserve_scalar_bits():
    x = numpy.lib.stride_tricks.sliding_window_view(numpy.random.default_rng(81).normal(size=1_007), 1_000)
    actual = svd_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([svd_entropy(row) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("shape", [(0, 100), (1, 100), (2, 0, 100), (2, 3, 4, 100)])
def test_batch_output_shape_including_empty_axes(shape):
    actual = svd_entropy(numpy.random.default_rng(1).normal(size=shape))
    assert isinstance(actual, numpy.ndarray)
    assert actual.shape == shape[:-1]
    assert actual.dtype == numpy.float64
    assert isinstance(svd_entropy(numpy.arange(100)), float)


@pytest.mark.parametrize("invalid", [numpy.nan, numpy.inf, -numpy.inf])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_nonfinite_error_names_first_full_index(invalid, layout):
    x = numpy.zeros((2, 3, 40))
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "reversed":
        x = x[..., ::-1]
    x[1, 1, 7] = invalid
    x[1, 2, 0] = numpy.nan
    with pytest.raises(ValueError) as error:
        svd_entropy(x)
    assert "finite" in str(error.value)
    assert "at index (1, 1, 7)" in str(error.value)


def test_nonfinite_scalar_input_names_time_index():
    with pytest.raises(ValueError, match="at index 7"):
        svd_entropy([0.0] * 7 + [numpy.inf])
