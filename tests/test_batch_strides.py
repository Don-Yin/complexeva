"""Retained-stride regression checks for the four float64 row-buffer batch kernels."""

import importlib

import numpy
import pytest

import complexeva

MEASURES = [
    ("hfd", "higuchi", "_hfd_rows"),
    ("dfa", "dfa", "_dfa_rows"),
    ("permutation_entropy", "permutation_entropy", "_PERMUTATION_ROWS"),
    ("sample_entropy", "sample_entropy", "_match_counts_rows"),
]


def assert_scalar_bits(measure, x):
    """Check every bit, including signed zero and NaN, without flattening the input."""
    actual = measure(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([measure(x[index]) for index in numpy.ndindex(x.shape[:-1])]).reshape(x.shape[:-1])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("metric, module_name, kernel_name", MEASURES)
@pytest.mark.parametrize("dtype", [numpy.bool_, numpy.int64, numpy.uint64, numpy.float32, numpy.float64])
@pytest.mark.parametrize("layout", ["C", "windows", "transposed_reversed"])
def test_batch_reaches_kernel_without_copying(monkeypatch, metric, module_name, kernel_name, dtype, layout):
    base = numpy.random.default_rng(25).integers(0, 21, size=(3, 2, 256)).astype(dtype)
    if layout == "C":
        x = base
    elif layout == "windows":
        x = numpy.lib.stride_tricks.sliding_window_view(base[0, 0], 128)[::17]
    else:
        x = base.transpose(1, 0, 2)[..., ::-2]
    x.flags.writeable = False
    before = x.copy()
    module = importlib.import_module(f"complexeva.{module_name}")
    kernels = getattr(module, kernel_name)
    kernel = kernels[3] if metric == "permutation_entropy" else kernels

    def checked(values, direct, *args):
        assert values is x
        copy_rows = direct is None if metric in ("hfd", "permutation_entropy") else args[-1]
        expected_copy = dtype != numpy.float64 or layout != "C"
        if metric in ("sample_entropy", "hfd", "permutation_entropy") and dtype == numpy.float64 and layout == "windows":
            expected_copy = False
        if metric == "dfa":
            # DFA's final flag selects flat logical access, not a copying workspace.
            expected_copy = x.ndim > 2 and not x.flags.c_contiguous
        assert copy_rows == expected_copy
        compiled = kernel(values, direct, *args)
        if not copy_rows:
            assert numpy.shares_memory(direct, x)
            kernel.py_func(values, direct, *args)  # Warm scalar dispatchers before guarding Python allocations.
            original_empty = numpy.empty

            def no_row_buffer(shape, *empty_args, **kwargs):
                if numpy.prod(shape) >= x.shape[-1]:
                    raise AssertionError("C-contiguous float64 must not allocate row-buffer storage")
                return original_empty(shape, *empty_args, **kwargs)

            with monkeypatch.context() as context:
                context.setattr(numpy, "empty", no_row_buffer)
                uncompiled = kernel.py_func(values, direct, *args)
            numpy.testing.assert_array_equal(compiled.view(numpy.uint64), uncompiled.view(numpy.uint64))
        elif metric in ("hfd", "permutation_entropy"):
            assert direct is None
        else:
            assert direct.size == 0
        return compiled

    if metric == "permutation_entropy":
        monkeypatch.setitem(kernels, 3, checked)
    else:
        monkeypatch.setattr(module, kernel_name, checked)
    assert_scalar_bits(getattr(complexeva, metric), x)
    numpy.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("metric, module_name, kernel_name", MEASURES)
@pytest.mark.parametrize("dtype", [numpy.float16, numpy.longdouble, ">f8"])
def test_kernel_unsupported_dtypes_keep_float64_semantics(metric, module_name, kernel_name, dtype):
    x = numpy.random.default_rng(26).normal(size=(2, 3, 256)).astype(dtype).transpose(1, 0, 2)[..., ::-2]
    x.flags.writeable = False
    assert_scalar_bits(getattr(complexeva, metric), x)


@pytest.mark.parametrize("metric, module_name, kernel_name", MEASURES)
@pytest.mark.parametrize("invalid", [numpy.nan, numpy.inf, -numpy.inf])
@pytest.mark.filterwarnings("error")
def test_nonfinite_index_follows_logical_order_with_transposed_negative_strides(metric, module_name, kernel_name, invalid):
    x = numpy.ones((3, 2, 256)).transpose(1, 0, 2)[..., ::-2]
    x[0, 2, 99] = invalid
    x[1, 0, 0] = invalid
    with pytest.raises(ValueError, match=r"finite.*index \(0, 2, 99\)"):
        getattr(complexeva, metric)(x)


def test_zero_tolerance_does_not_skip_finite_validation():
    x = numpy.ones((2, 3, 100))[:, ::-1, ::-1]
    x[1, 1, 7] = numpy.inf
    with pytest.raises(ValueError, match=r"finite.*index \(1, 1, 7\)"):
        complexeva.sample_entropy(x, tolerance=0.0)
