import re

import numpy
import pytest

from complexeva import (
    dfa,
    hfd,
    hfd_hilbert,
    lempel_ziv,
    lempel_ziv_hilbert,
    lyapunov_exponent,
    permutation_entropy,
    sample_entropy,
)
from complexeva._validation import as_finite_1d, check_positive_int

# Every numeric measure; all name their signal argument x.
MEASURES = [hfd, hfd_hilbert, lempel_ziv, lempel_ziv_hilbert, lyapunov_exponent, sample_entropy, permutation_entropy, dfa]


@pytest.mark.parametrize("measure", MEASURES)
@pytest.mark.parametrize(
    "data, bad_value",
    [
        ([0.0, numpy.nan], "nan"),
        ([0.0, numpy.inf], "inf"),
        ([0.0, -numpy.inf], "-inf"),
        (["bad"], "bad"),
        ([1 + 2j], "1.+2.j"),
        ([[1.0], [2.0, 3.0]], "[[1.0], [2.0, 3.0]]"),
        (1.0, "shape ()"),
    ],
)
def test_invalid_signal_names_parameter_and_value(measure, data, bad_value):
    with pytest.raises(ValueError) as error:
        measure(data)
    assert "x must" in str(error.value)
    assert bad_value in str(error.value)


@pytest.mark.parametrize("measure", [m for m in MEASURES if m is not lempel_ziv])
def test_empty_signal_names_parameter_and_size(measure):
    with pytest.raises(ValueError) as error:
        measure(numpy.array([]))
    assert re.search(r"\bx\b", str(error.value))
    assert "0" in str(error.value)


@pytest.mark.parametrize("measure", MEASURES)
def test_arraylike_input_matches_float64_array(measure):
    data = numpy.random.default_rng(19).standard_normal(100)
    numpy.testing.assert_array_equal(measure(data.tolist()), measure(data))


def test_nonfinite_error_names_value_and_index():
    with pytest.raises(ValueError, match="signal.*inf.*index 1"):
        as_finite_1d([0, numpy.inf], name="signal")


def test_shape_error_names_shape():
    with pytest.raises(ValueError, match=r"signal.*shape \(2, 3\)"):
        as_finite_1d(numpy.ones((2, 3)), name="signal")


def test_empty_series_leaves_length_validation_to_caller():
    values = as_finite_1d([])
    assert values.shape == (0,)
    assert values.dtype == numpy.float64


@pytest.mark.parametrize("value", [True, numpy.bool_(True), 0, -1, 1.0, "1", numpy.nan, None])
def test_positive_integer_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="window"):
        check_positive_int(value, "window")


def test_positive_integer_validation_returns_builtin_integer():
    value = check_positive_int(numpy.uint64(4), "window", minimum=4)
    assert value == 4 and type(value) is int
    with pytest.raises(ValueError, match="window.*>= 4.*3"):
        check_positive_int(3, "window", minimum=4)


@pytest.mark.parametrize("stride", [1, 2, -1])
def test_first_nonfinite_index_and_message(stride):
    x = numpy.array([1.0, numpy.inf, numpy.nan, -numpy.inf, 2.0])[::stride]
    first = int(numpy.flatnonzero(~numpy.isfinite(x))[0])
    with pytest.raises(ValueError) as error:
        as_finite_1d(x, "signal")
    assert str(error.value) == f"signal must contain only finite values, got {x[first]!r} at index {first}"


@pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32, numpy.int16, numpy.bool_, ">f8"])
def test_conversion_and_readonly_views(dtype):
    x = numpy.arange(100).astype(dtype)[::-2]
    x.flags.writeable = False
    actual = as_finite_1d(x)
    assert actual.dtype == numpy.float64
    numpy.testing.assert_array_equal(actual, numpy.asarray(x, dtype=numpy.float64))


def test_unaligned_array():
    x = numpy.ndarray(10, dtype=numpy.float64, buffer=bytearray(81), offset=1)
    x[:] = numpy.arange(10)
    assert not x.flags.aligned
    numpy.testing.assert_array_equal(as_finite_1d(x), x)
