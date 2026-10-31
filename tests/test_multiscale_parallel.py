import importlib

import numpy as np
import pytest

from complexeva._validation import unit_stride_rows
from complexeva.multiscale_entropy import _multiscale_rows, multiscale_entropy


@pytest.mark.parametrize("modified", [False, True])
@pytest.mark.parametrize("chunks", [1, 2, 3, 8])
def test_uneven_row_chunks_match_scalar_bits(chunks, modified):
    values = np.random.default_rng(4040).normal(size=(13, 257)).astype(np.float32)
    factors = np.array([5, 2, 1, 2])
    result = _multiscale_rows(values, unit_stride_rows(values), factors, 2, np.full(13, 0.7), modified, chunks)
    expected = np.array(
        [multiscale_entropy(row, factors, r=0.7, method="modified" if modified else "coarse") for row in values]
    )
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("method", ["coarse", "modified"])
@pytest.mark.parametrize("scales", [[1], [1, 2], [2, 5], [20]])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_concurrent_tree_and_coarse_workspace_is_bounded(monkeypatch, method, scales, dtype):
    module = importlib.import_module("complexeva.multiscale_entropy")
    monkeypatch.setattr(module, "get_num_threads", lambda: 16)
    calls = []

    def capture(values, direct, factors, order, tolerances, modified, chunks):
        calls.append(chunks)
        return np.empty((len(tolerances), len(factors)))

    monkeypatch.setattr(module, "_multiscale_rows", capture)
    values = np.broadcast_to(np.zeros(100000, dtype=dtype), (64, 100000))
    multiscale_entropy(values, scales, r=0, method=method)
    minimum = min(scales)
    length = 100000 - minimum + 1 if method == "modified" else 100000 // minimum
    templates = length - 2 * (minimum if method == "modified" else 1)
    coarse_scale = min((scale for scale in scales if scale > 1), default=100001)
    coarse_n = max(0, 100001 - coarse_scale) if method == "modified" else 100000 // coarse_scale
    workspace = 8 * (coarse_n + (100000 if dtype == np.float32 else 0))
    if templates >= module._TREE_MIN_TEMPLATES:
        workspace += module._tree_workspace_bytes(templates, 3)
    expected = min(16, max(1, module._TREE_BATCH_BYTES // workspace))
    assert calls == ([expected] if expected > 1 else [])
    assert expected * workspace <= module._TREE_BATCH_BYTES


@pytest.mark.parametrize("method", ["coarse", "modified"])
def test_single_worker_budget_avoids_parallel_scheduler(monkeypatch, method):
    module = importlib.import_module("complexeva.multiscale_entropy")
    monkeypatch.setattr(module, "_TREE_BATCH_BYTES", 1)

    def unexpected_parallel(*args):
        raise AssertionError("one worker must use the serial schedule")

    monkeypatch.setattr(module, "_multiscale_rows", unexpected_parallel)
    result = multiscale_entropy(np.ones((3, 100)), [1, 2], r=0, method=method)
    assert np.isnan(result).all()
    assert np.isnan(multiscale_entropy(np.ones(100), [1, 2], r=0, method=method)).all()
