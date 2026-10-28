import re

import numpy
import pytest

from complexeva import lyapunov_exponent


def logistic(n, x0=0.3):
    x = numpy.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    return x


def henon(n):
    x, y, out = 0.1, 0.1, numpy.empty(n)
    for i in range(n):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        out[i] = x
    return out


def test_logistic_map_is_ln2():
    assert lyapunov_exponent(logistic(3000), m=1, trajectory_len=8) == pytest.approx(numpy.log(2), abs=0.03)


def test_henon_map_known_value():
    assert lyapunov_exponent(henon(3000), m=2, trajectory_len=8) == pytest.approx(0.419, abs=0.03)


def test_regular_signal_is_about_zero():
    x = numpy.sin(numpy.linspace(0, 40 * numpy.pi, 3000))
    assert abs(lyapunov_exponent(x)) < 0.01


def test_scale_and_offset_invariant():
    x = logistic(2000)
    a = lyapunov_exponent(x, m=1, trajectory_len=8)
    assert lyapunov_exponent(1000 * x + 5, m=1, trajectory_len=8) == pytest.approx(a, rel=1e-9)


def test_dt_scales_result():
    x = logistic(2000)
    assert lyapunov_exponent(x, m=1, trajectory_len=8, dt=0.5) == pytest.approx(2 * lyapunov_exponent(x, m=1, trajectory_len=8))


def test_too_short_raises():
    with pytest.raises(ValueError, match=r"len\(data\)=10.*min_tsep=5"):
        lyapunov_exponent(numpy.arange(10.0), min_tsep=5)


@pytest.mark.parametrize("data", [numpy.array(1.0), numpy.ones((2, 10))])
def test_invalid_shape_names_bad_shape(data):
    with pytest.raises(ValueError, match=re.escape(str(data.shape))):
        lyapunov_exponent(data)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("tau", 0),
        ("tau", -1),
        ("tau", 1.5),
        ("tau", True),
        ("m", 0),
        ("m", -1),
        ("m", 1.5),
        ("m", True),
        ("trajectory_len", 0),
        ("trajectory_len", 1),
        ("trajectory_len", 2.5),
        ("trajectory_len", True),
        ("fit_offset", -1),
        ("fit_offset", 19),
        ("fit_offset", 1.5),
        ("fit_offset", True),
        ("min_tsep", -1),
        ("min_tsep", 1.5),
        ("min_tsep", True),
        ("dt", 0),
        ("dt", -1.0),
        ("dt", float("nan")),
        ("dt", float("inf")),
        ("dt", True),
        ("dt", "1"),
    ],
)
def test_invalid_parameter_names_bad_value(parameter, value):
    with pytest.raises(ValueError) as error:
        lyapunov_exponent(numpy.arange(100.0), **{parameter: value})
    assert parameter in str(error.value)
    assert repr(value) in str(error.value)


def test_constant_default_window_names_bad_value():
    with pytest.raises(ValueError, match="min_tsep.*constant.*1.0"):
        lyapunov_exponent(numpy.ones(100))


def test_constant_with_explicit_window_returns_negative_infinity():
    assert lyapunov_exponent(numpy.ones(100), min_tsep=0) == -numpy.inf


def test_numpy_integer_parameters_are_valid():
    x = logistic(100)
    expected = lyapunov_exponent(x, tau=1, m=2, trajectory_len=4, fit_offset=0, min_tsep=2)
    assert (
        lyapunov_exponent(
            x,
            tau=numpy.int64(1),
            m=numpy.int64(2),
            trajectory_len=numpy.int64(4),
            fit_offset=numpy.int64(0),
            min_tsep=numpy.int64(2),
        )
        == expected
    )
