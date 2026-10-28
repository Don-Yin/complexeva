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
    with pytest.raises(ValueError):
        lyapunov_exponent(numpy.arange(10.0), min_tsep=5)
