import subprocess
import sys

import numpy
import pytest

import complexeva as cpu


@pytest.fixture(scope="module")
def tensor_backend():
    torch = pytest.importorskip("torch")
    from complexeva import gpu

    return torch, gpu


@pytest.fixture(params=["cpu", "mps", "cuda"])
def device(request, tensor_backend):
    torch, _ = tensor_backend
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if not available[request.param]:
        pytest.skip(f"PyTorch {request.param} device not available")
    return request.param


def test_cpu_import_does_not_require_torch():
    command = """
import sys
sys.modules['torch'] = None
import complexeva
assert complexeva.surrogate_test(max, [1, 2], 2, seed=0).pvalue == 1.0
try:
    import complexeva.gpu
except ImportError as error:
    assert "complexeva[gpu]" in str(error), error
else:
    raise AssertionError('optional GPU module imported without torch')
"""
    subprocess.run([sys.executable, "-c", command], check=True)


def test_gpu_public_api(tensor_backend):
    _, gpu = tensor_backend
    assert set(gpu.__all__) == {"SpectrumShuffleResult", "hfd", "spectrum_shuffle_test"}
    assert not hasattr(gpu, "sample_entropy") and not hasattr(gpu, "lempel_ziv")


@pytest.mark.parametrize(("n", "k_max"), [(20, 10), (87, 16)])
def test_higuchi_matches_cpu_including_zero_curve_lengths(tensor_backend, device, n, k_max):
    _, gpu = tensor_backend
    rng = numpy.random.default_rng(2)
    data = numpy.stack((numpy.arange(n), rng.normal(size=n).cumsum(), numpy.ones(n), (-1.0) ** numpy.arange(n)))
    x = data.reshape(2, 2, n)[..., ::-1]
    x.flags.writeable = False
    actual = gpu.hfd(x, k_max, device=device)
    tol = 2e-6 if device == "mps" else 1e-12
    assert actual.shape == (2, 2) and actual.dtype == numpy.float64
    numpy.testing.assert_allclose(actual, cpu.hfd(x, k_max), atol=tol, rtol=0)
    assert isinstance(gpu.hfd(x[0, 0], k_max, device=device), float)


@pytest.mark.parametrize("n", [64, 65])
def test_phase_shuffle_preserves_each_fourier_magnitude(tensor_backend, device, n):
    torch, _ = tensor_backend
    from complexeva.gpu._shuffle import _phase_shuffles

    dtype = torch.float32 if device == "mps" else torch.float64
    x = numpy.random.default_rng(7).normal(size=(2, n)) + numpy.array([1.0, -3.0])[:, None]
    rows = torch.from_numpy(x).to(dtype=dtype).to(device)
    copies = _phase_shuffles(rows, 5, numpy.random.default_rng(5)).cpu().numpy()
    spectrum = numpy.abs(numpy.fft.rfft(copies))
    baseline = numpy.broadcast_to(numpy.abs(numpy.fft.rfft(x))[:, None], spectrum.shape)
    tol = 1e-4 if device == "mps" else 1e-12
    numpy.testing.assert_allclose(spectrum, baseline, atol=tol, rtol=0)
    means = numpy.broadcast_to(x.mean(axis=-1, keepdims=True), copies.shape[:-1])
    numpy.testing.assert_allclose(copies.mean(axis=-1), means, atol=tol, rtol=0)
    assert not numpy.allclose(copies[:, 0], x)
    assert not numpy.allclose(copies[:, 0], copies[:, 1])


def test_shuffle_scores_and_pvalue(tensor_backend, device):
    torch, gpu = tensor_backend
    from complexeva.gpu._shuffle import _phase_shuffles

    x = numpy.random.default_rng(13).normal(size=(2, 1, 60)).cumsum(axis=-1)
    # Rounds to a constant on Metal; the observed CPU score still uses the original samples.
    x[1, 0] = 1e8 + numpy.tile([0.0, 1.0], 30)
    result = gpu.spectrum_shuffle_test(x, order=1, n_shuffles=7, seed=5, device=device)
    observed = numpy.asarray(result.observed)
    pvalue = numpy.asarray(result.pvalue)
    numpy.testing.assert_array_equal(observed, cpu.sample_entropy(x, order=1))
    assert numpy.isfinite(observed).all()
    assert result.shuffled.shape == (2, 1, 7)
    dtype = torch.float32 if device == "mps" else torch.float64
    rows = torch.from_numpy(x.reshape(2, 60)).to(dtype=dtype).to(device)
    copies = _phase_shuffles(rows, 7, numpy.random.default_rng(5)).cpu().numpy()
    expected = numpy.asarray(cpu.sample_entropy(copies, order=1)).reshape(2, 1, 7)
    numpy.testing.assert_array_equal(result.shuffled, expected)
    expected_p = (1 + (expected <= observed[..., None]).sum(axis=-1)) / 8
    expected_p[numpy.isnan(expected).any(axis=-1)] = numpy.nan
    numpy.testing.assert_array_equal(pvalue, expected_p)
    repeat = gpu.spectrum_shuffle_test(x[0, 0], order=1, n_shuffles=7, seed=5, device=device)
    assert isinstance(repeat.observed, float) and isinstance(repeat.pvalue, float)
    numpy.testing.assert_array_equal(repeat.shuffled, result.shuffled[0, 0])


def test_torch_inputs_and_validation(tensor_backend, device):
    torch, gpu = tensor_backend
    rows = torch.arange(50, dtype=torch.float32, device=device).reshape(2, 25).requires_grad_()
    expected_hfd = cpu.hfd(rows.detach().cpu().numpy())
    numpy.testing.assert_allclose(gpu.hfd(rows), expected_hfd, atol=2e-6, rtol=0)
    numpy.testing.assert_allclose(gpu.hfd(rows, device="cpu"), expected_hfd, atol=1e-12, rtol=0)
    result = gpu.spectrum_shuffle_test(rows, n_shuffles=2, tolerance=2.5, seed=4)
    numpy.testing.assert_array_equal(result.observed, cpu.sample_entropy(rows.detach().cpu().numpy(), tolerance=2.5))
    assert numpy.asarray(gpu.hfd(numpy.empty((0, 20)), device=device)).shape == (0,)
    empty = gpu.spectrum_shuffle_test(numpy.empty((2, 0, 20)), n_shuffles=5, device=device)
    assert empty.shuffled.shape == (2, 0, 5)
    bad = rows.detach().clone()
    bad[1, 3] = float("inf")
    with pytest.raises(ValueError, match=r"finite.*index \(1, 3\)"):
        gpu.hfd(bad)
    with pytest.raises(ValueError, match="order"):
        gpu.spectrum_shuffle_test(rows, order=True, device=device)
    with pytest.raises(ValueError, match="tolerance"):
        gpu.spectrum_shuffle_test(rows, tolerance=-1, device=device)
    with pytest.raises(ValueError, match="k_max"):
        gpu.hfd(rows, k_max=13, device=device)
    with pytest.raises(ValueError, match="n_shuffles"):
        gpu.spectrum_shuffle_test(rows, n_shuffles=0, device=device)
