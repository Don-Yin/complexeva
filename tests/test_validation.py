import numpy
import pytest

from complexeva import dfa, permutation_entropy, sample_entropy
from complexeva._validation import as_finite_1d, check_positive_int


@pytest.mark.parametrize("measure", [sample_entropy, permutation_entropy, dfa])
@pytest.mark.parametrize("x", [1, [[1, 2], [3, 4]], [0, numpy.nan], [numpy.inf], [1j], ["1", "2"], [[1], [2, 3]]])
def test_new_measures_reject_invalid_series(measure, x):
    with pytest.raises(ValueError, match="x"):
        measure(x)


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


@pytest.mark.parametrize("dtype", [numpy.int32, numpy.float32, numpy.bool_])
def test_numeric_series_are_converted_to_float64(dtype):
    values = as_finite_1d(numpy.array([0, 1], dtype=dtype))
    assert values.dtype == numpy.float64
    numpy.testing.assert_array_equal(values, [0, 1])


@pytest.mark.parametrize("value", [True, numpy.bool_(True), 0, -1, 1.0, "1", numpy.nan, None])
def test_positive_integer_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="window"):
        check_positive_int(value, "window")


def test_positive_integer_validation_returns_builtin_integer():
    value = check_positive_int(numpy.uint64(4), "window", minimum=4)
    assert value == 4 and type(value) is int
    with pytest.raises(ValueError, match="window.*>= 4.*3"):
        check_positive_int(3, "window", minimum=4)
