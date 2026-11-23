from functools import partial
from typing import Any

import numpy
import pytest

from complexeva import SurrogateTestResult, iaaft_surrogates, phase_randomized_surrogates, sample_entropy, surrogate_test


@pytest.mark.parametrize("samples", [63, 64])
def test_phase_preserves_spectrum_and_real_bins_in_strided_batch(samples):
    rng = numpy.random.default_rng(37)
    x = rng.normal(size=(2, 3, 2 * samples))[..., ::-2] + numpy.arange(6).reshape(2, 3, 1)
    x = x[..., ::-1]
    x.flags.writeable = False
    copies = phase_randomized_surrogates(x, 5, seed=12)
    assert copies.shape == (2, 3, 5, samples) and copies.dtype == numpy.float64
    reference = numpy.fft.rfft(x)[..., None, :]
    spectrum = numpy.fft.rfft(copies)
    reference = numpy.broadcast_to(reference, spectrum.shape)
    numpy.testing.assert_allclose(numpy.abs(spectrum), numpy.abs(reference), atol=1e-12, rtol=0)
    numpy.testing.assert_allclose(spectrum[..., 0], reference[..., 0], atol=1e-12, rtol=0)
    if samples % 2 == 0:
        numpy.testing.assert_allclose(spectrum[..., -1], reference[..., -1], atol=1e-12, rtol=0)
    assert not numpy.allclose(copies[..., 0, :], x)
    assert not numpy.allclose(copies[..., 0, :], copies[..., 1, :])
    numpy.testing.assert_array_equal(copies, phase_randomized_surrogates(x, 5, seed=12))
    numpy.testing.assert_array_equal(copies[0, 0], phase_randomized_surrogates(x[0, 0], 5, seed=12))
    assert not numpy.allclose(copies, phase_randomized_surrogates(x, 5, seed=13))


def test_phase_scales_fft_for_large_finite_signals():
    x = numpy.random.default_rng(3).normal(size=61)
    copies = phase_randomized_surrogates(x * 1e300, 6, seed=1)
    spectrum = numpy.abs(numpy.fft.rfft(copies / 1e300))
    reference = numpy.broadcast_to(numpy.abs(numpy.fft.rfft(x)), spectrum.shape)
    numpy.testing.assert_allclose(spectrum, reference, atol=1e-12, rtol=0)


def test_iaaft_preserves_tied_samples_while_improving_spectral_match():
    rng = numpy.random.default_rng(4)
    noise = rng.normal(size=(2, 128))
    x = numpy.round(numpy.exp((noise + numpy.roll(noise, 1, axis=-1)) / 2), 1)
    copies = iaaft_surrogates(x, 4, seed=15)
    numpy.testing.assert_array_equal(numpy.sort(copies), numpy.broadcast_to(numpy.sort(x)[:, None], copies.shape))
    numpy.testing.assert_array_equal(copies, iaaft_surrogates(x, 4, seed=15))
    assert not numpy.array_equal(copies[0, 0], copies[0, 1])

    def spectral_error(rows):
        target = numpy.abs(numpy.fft.rfft(x)[:, None, 1:])
        magnitude = numpy.abs(numpy.fft.rfft(rows)[..., 1:])
        return numpy.linalg.norm(magnitude - target, axis=-1) / numpy.linalg.norm(target, axis=-1)

    one_iteration = iaaft_surrogates(x, 4, seed=15, max_iter=1)
    improved = spectral_error(copies)
    assert (improved < 0.20).all()
    assert (improved <= spectral_error(one_iteration) + 1e-12).all()
    assert (improved < spectral_error(one_iteration) - 1e-3).any()


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("method", ["phase", "iaaft"])
def test_surrogate_test_matches_direct_scores_and_rank_counts(method, alternative):
    x = numpy.random.default_rng(6).normal(size=(2, 3, 32))
    generator = phase_randomized_surrogates if method == "phase" else iaaft_surrogates
    copies = generator(x, 8, seed=0)

    # A scalar-only measure, not assumed to support vectorised calls.
    def measure(row):
        assert row.ndim == 1 and row.dtype == numpy.float64
        return row[0]

    result = surrogate_test(measure, x, 8, seed=0, method=method, alternative=alternative)
    assert isinstance(result, SurrogateTestResult)
    scores = copies[..., 0]
    observed = x[..., 0]
    mean = scores.mean(axis=-1)
    lower = (1 + (scores <= observed[..., None]).sum(axis=-1)) / 9
    upper = (1 + (scores >= observed[..., None]).sum(axis=-1)) / 9
    expected = {"less": lower, "greater": upper, "two-sided": numpy.minimum(1, 2 * numpy.minimum(lower, upper))}
    for actual, baseline in zip(result, (observed, mean, observed - mean, expected[alternative]), strict=True):
        assert isinstance(actual, numpy.ndarray) and actual.shape == (2, 3) and actual.dtype == numpy.float64
        numpy.testing.assert_array_equal(actual, baseline)
    scalar = surrogate_test(measure, x[0, 0], 8, seed=0, method=method, alternative=alternative)
    assert all(type(value) is float for value in scalar)
    assert scalar == tuple(numpy.asarray(array)[0, 0] for array in result)


def test_constant_short_and_empty_batches():
    x = numpy.full(31, 1 / 3)
    for generator in (phase_randomized_surrogates, iaaft_surrogates):
        numpy.testing.assert_array_equal(generator(x, 5, seed=19), numpy.broadcast_to(x, (5, 31)))
        assert generator(numpy.empty((2, 0, 31)), 5, seed=19).shape == (2, 0, 5, 31)
    numpy.testing.assert_array_equal(phase_randomized_surrogates([1, 2], 2, seed=19), [[1, 2], [1, 2]])
    result = surrogate_test(lambda _: pytest.fail("measure ran on an empty batch"), numpy.empty((2, 0, 31)), 5)
    assert all(isinstance(field, numpy.ndarray) and field.shape == (2, 0) for field in result)


def test_ties_and_undefined_scores():
    x = numpy.ones(31)
    for alternative in ("less", "greater", "two-sided"):
        result = surrogate_test(numpy.sum, x, 3, alternative=alternative, seed=0)
        assert result == (31.0, 31.0, 0.0, 1.0)
    undefined = surrogate_test(partial(sample_entropy, order=1), x, 3, seed=0)
    assert numpy.isnan(numpy.asarray(undefined)).all()
    result = surrogate_test(lambda _: float("inf"), x, 3, alternative="less")
    assert result.observed == result.surrogate_mean == float("inf")
    assert numpy.isnan(result.delta) and result.pvalue == 1.0
    # A NaN surrogate score alone is enough to invalidate the row's p-value.
    calls = iter([1.0, 2.0, float("nan")])
    result = surrogate_test(lambda _: next(calls), x, 2)
    assert result.observed == 1.0 and numpy.isnan(numpy.asarray(result[1:])).all()


@pytest.mark.parametrize(
    ("value", "message"),
    [(1.0, "time axis"), ([2.0], "at least 2 samples"), ([1.0, 2j], "real numbers"), ([[1.0, numpy.inf]], r"index \(0, 1\)")],
)
def test_phase_rejects_invalid_signals(value, message):
    with pytest.raises(ValueError, match=message):
        phase_randomized_surrogates(value, 2)


def test_surrogate_options_and_measure_validation():
    x = numpy.arange(15)
    with pytest.raises(ValueError, match="n must be an integer"):
        phase_randomized_surrogates(x, True)
    with pytest.raises(ValueError, match="max_iter"):
        iaaft_surrogates(x, 2, max_iter=0)
    bad_option: Any = "unknown"
    not_callable: Any = 1
    with pytest.raises(ValueError, match="method"):
        surrogate_test(numpy.max, x, method=bad_option)
    with pytest.raises(ValueError, match="alternative"):
        surrogate_test(numpy.max, x, alternative=bad_option)
    with pytest.raises(TypeError, match="callable"):
        surrogate_test(not_callable, x)
    with pytest.raises(ValueError, match="one real scalar"):
        surrogate_test(lambda row: row[:2], x)
