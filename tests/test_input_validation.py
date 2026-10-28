import numpy
import pytest

from complexeva import hfd, hfd_hilbert, lempel_ziv, lempel_ziv_hilbert, lyapunov_exponent


@pytest.mark.parametrize(
    "measure, name",
    [
        (hfd, "X"),
        (hfd_hilbert, "X"),
        (lempel_ziv, "time_series"),
        (lempel_ziv_hilbert, "time_series"),
        (lyapunov_exponent, "data"),
    ],
)
@pytest.mark.parametrize(
    "data, bad_value",
    [
        ([0.0, numpy.nan], "nan"),
        ([0.0, numpy.inf], "inf"),
        ([0.0, -numpy.inf], "-inf"),
        (["bad"], "bad"),
        ([1 + 2j], "1.+2.j"),
        ([[1.0], [2.0, 3.0]], "[[1.0], [2.0, 3.0]]"),
    ],
)
def test_invalid_numeric_data_names_parameter_and_value(measure, name, data, bad_value):
    with pytest.raises(ValueError) as error:
        measure(data)
    assert name in str(error.value)
    assert bad_value in str(error.value)


@pytest.mark.parametrize(
    "measure, name", [(hfd, "X"), (hfd_hilbert, "X"), (lempel_ziv_hilbert, "time_series"), (lyapunov_exponent, "data")]
)
def test_empty_series_names_parameter_and_size(measure, name):
    with pytest.raises(ValueError) as error:
        measure(numpy.array([]))
    assert name in str(error.value)
    assert "0" in str(error.value)


def test_lyapunov_single_sample_names_length():
    with pytest.raises(ValueError, match=r"len\(data\)=1"):
        lyapunov_exponent([1.0])


@pytest.mark.parametrize("measure", [hfd, hfd_hilbert, lempel_ziv, lempel_ziv_hilbert, lyapunov_exponent])
def test_arraylike_input_matches_float64_array(measure):
    data = numpy.random.default_rng(19).standard_normal(100)
    assert measure(data.tolist()) == measure(data)


@pytest.mark.parametrize("data", ["", []])
@pytest.mark.parametrize("method", ["mean", "median", "diff"])
def test_lempel_ziv_empty_sequence_preserves_counts(data, method):
    assert lempel_ziv(data, method=method) == 1
    assert lempel_ziv(data, method=method, normalize=True) == 0.0


def test_lempel_ziv_single_difference_preserves_counts():
    assert lempel_ziv([1.0], method="diff") == 1
    assert lempel_ziv([1.0], method="diff", normalize=True) == 0.0


def test_empty_numeric_series_still_validates_method():
    with pytest.raises(ValueError, match="method.*'mode'"):
        lempel_ziv([], method="mode")
