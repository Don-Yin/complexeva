import importlib
import itertools

import numpy
import pytest

from complexeva import sample_entropy
from complexeva._sampen_tree import _build, _count, _tree_match_count, _tree_workspace_bytes
from complexeva.sample_entropy import _TREE_MAX_ORDER, _TREE_MIN_TEMPLATES, _diagonal_match_counts, _match_counts


def test_known_binary_templates_match_antropy_and_nolds():
    # antropy 0.2.3 and nolds 0.6.3, order=2, tolerance=0.5, strict Chebyshev.
    # Five length-2 pairs match, but only two still match at length 3.
    x = numpy.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    assert sample_entropy(x, tolerance=0.5) == pytest.approx(numpy.log(5 / 2), abs=1e-14)


def test_fixed_noise_matches_antropy_and_nolds():
    # antropy 0.2.3 defaults; nolds 0.6.3 with tolerance=0.2*std(x, ddof=0).
    x = numpy.random.default_rng(0).standard_normal(2048)
    assert sample_entropy(x) == pytest.approx(2.1844831289343847, abs=1e-14)


def test_minimum_length_has_one_pair():
    assert sample_entropy([1, 1, 1, 1], tolerance=0.5) == 0.0


def test_strict_tolerance_excludes_boundary_matches():
    x = [0, 0, 1, 2]
    assert numpy.isinf(sample_entropy(x, order=1, tolerance=1.0))
    assert sample_entropy(x, order=1, tolerance=numpy.nextafter(1.0, numpy.inf)) == pytest.approx(numpy.log(3 / 2))


def test_no_matching_templates_is_nan():
    assert numpy.isnan(sample_entropy(numpy.arange(10), tolerance=0.5))
    assert numpy.isnan(sample_entropy(numpy.ones(10)))
    assert numpy.isnan(sample_entropy(numpy.ones(10), tolerance=0))


def test_constant_and_periodic_signals_have_zero_entropy_at_positive_tolerance():
    assert sample_entropy(numpy.ones(10), tolerance=0.5) == 0.0
    assert sample_entropy(numpy.tile([0, 1], 32), tolerance=0.5) == 0.0


def test_matches_without_extensions_give_infinity():
    assert numpy.isinf(sample_entropy([0, 0, 1], order=1, tolerance=0.5))


def test_default_tolerance_and_scale_offset_invariance():
    x = numpy.random.default_rng(1).standard_normal(512)
    expected = sample_entropy(x, tolerance=0.2 * numpy.std(x, ddof=0))
    assert sample_entropy(x) == expected
    assert sample_entropy(7 * x + 42) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("readonly", [False, True])
def test_noncontiguous_and_readonly_inputs(readonly):
    x = numpy.random.default_rng(2).standard_normal(600)[::2]
    x.setflags(write=not readonly)
    assert sample_entropy(x) == sample_entropy(x.copy())


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_short_series_rejected(length):
    with pytest.raises(ValueError, match="samples"):
        sample_entropy(numpy.ones(length))


@pytest.mark.parametrize("order", [0, -1, 1.5, True, numpy.float64(2), numpy.nan])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="order"):
        sample_entropy(numpy.ones(10), order=order)


@pytest.mark.parametrize("tolerance", [-1, numpy.nan, numpy.inf, True, "0.2", 1j, 10**400])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        sample_entropy(numpy.ones(10), tolerance=tolerance)


def test_numpy_integer_order_is_accepted():
    assert sample_entropy(numpy.ones(10), order=numpy.int64(2), tolerance=numpy.float64(0.5)) == 0.0


def literal_counts(x, order, tolerance):
    """Evaluate the public definition independently, without spatial pruning."""
    counts = [0, 0]
    for i, j in itertools.combinations(range(len(x) - order), 2):
        if all(abs(x[i + k] - x[j + k]) < tolerance for k in range(order)):
            counts[0] += 1
            counts[1] += abs(x[i + order] - x[j + order]) < tolerance
    return tuple(counts)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [9, 17, 33, 65])
@pytest.mark.parametrize("tolerance", [numpy.nextafter(1.0, 0.0), 1.0, numpy.nextafter(1.0, numpy.inf), 3.0])
def test_spatial_counts_equal_literal_definition(order, length, tolerance):
    x = numpy.random.default_rng(length).integers(-3, 4, length).astype(float)
    expected = literal_counts(x, order, tolerance)
    actual = tuple(_tree_match_count(x, dimension, length - order, tolerance) for dimension in (order, order + 1))
    assert actual == expected
    assert _diagonal_match_counts(x, order, tolerance) == expected


@pytest.mark.parametrize("scale", [numpy.nextafter(0.0, 1.0), 1e-300, 1.0, 1e300])
def test_extreme_magnitudes_and_exact_radius(scale):
    x = numpy.tile(numpy.array([0.0, 0.0, 1.0, -1.0, 2.0, 1.0]) * scale, 7)
    for tolerance in (scale, numpy.nextafter(scale, numpy.inf)):
        for order in (1, 2, 3):
            expected = literal_counts(x, order, tolerance)
            assert _diagonal_match_counts(x, order, tolerance) == expected
            assert tuple(_tree_match_count(x, d, len(x) - order, tolerance) for d in (order, order + 1)) == expected


def test_overflowing_distances_do_not_match():
    x = numpy.tile([numpy.finfo(float).max, -numpy.finfo(float).max, 0.0], 11)
    with numpy.errstate(over="ignore"):
        expected = literal_counts(x, 2, numpy.finfo(float).max)
    assert _diagonal_match_counts(x, 2, numpy.finfo(float).max) == expected
    assert tuple(_tree_match_count(x, d, len(x) - 2, numpy.finfo(float).max) for d in (2, 3)) == expected


@pytest.mark.parametrize("order", [1, 2, _TREE_MAX_ORDER - 1, _TREE_MAX_ORDER, _TREE_MAX_ORDER + 1, 17])
def test_dispatch_boundaries_match_diagonal_reference(order):
    for n_templates in (_TREE_MIN_TEMPLATES - 1, _TREE_MIN_TEMPLATES, _TREE_MIN_TEMPLATES + 1):
        x = numpy.random.default_rng(10 + order).normal(size=n_templates + order)
        assert _match_counts(x, order, 0.2) == _diagonal_match_counts(x, order, 0.2)


def test_wide_pair_counts_exclude_self_matches():
    n, order = 100_000, 2
    x = numpy.ones(n)
    expected = (n - order) * (n - order - 1) // 2
    assert expected > numpy.iinfo(numpy.int32).max
    assert _match_counts(x, order, 0.1) == (expected, expected)
    assert sample_entropy(x, tolerance=0.1) == 0.0
    assert numpy.isnan(sample_entropy(x))


def test_tree_preserves_nan_and_infinite_results():
    assert numpy.isnan(sample_entropy(numpy.arange(4100), tolerance=0.5))
    # The only duplicate length-two prefix has a nonmatching extension.
    x = numpy.arange(4100, dtype=float)
    x[2000:2003] = [0.0, 1.0, -1.0]
    assert numpy.isinf(sample_entropy(x, tolerance=0.5))


@pytest.mark.parametrize("order", [1, 2, _TREE_MAX_ORDER])
def test_tree_matches_diagonal_on_tied_and_clustered_series(order):
    # Identical templates and tight clusters exercise whole-block counting and the leaf-pair checks.
    rng = numpy.random.default_rng(order)
    tied = rng.integers(0, 3, 2 * _TREE_MIN_TEMPLATES).astype(float)
    clustered = rng.choice([-1.0, 1.0], 2 * _TREE_MIN_TEMPLATES) + 0.05 * rng.standard_normal(2 * _TREE_MIN_TEMPLATES)
    for x, tolerance in ((tied, 1.5), (clustered, 0.2)):
        assert _match_counts(x, order, tolerance) == _diagonal_match_counts(x, order, tolerance)


@pytest.mark.parametrize("shape", [(6, 261), (2, 3, 1000), (1, 2, 1, 4100)])
@pytest.mark.parametrize("order", [1, 2, 4, 5])
@pytest.mark.parametrize("tolerance", [None, 0.0, 0.2, 1.0])
def test_batch_is_bitwise_equal_to_1d_calls(shape, order, tolerance):
    x = numpy.random.default_rng(4).normal(size=shape)
    actual = sample_entropy(x, order=order, tolerance=tolerance)
    expected = numpy.array([sample_entropy(row, order=order, tolerance=tolerance) for row in x.reshape(-1, shape[-1])]).reshape(
        shape[:-1]
    )
    assert isinstance(actual, numpy.ndarray)
    assert actual.shape == shape[:-1]
    assert actual.dtype == numpy.float64
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("layout", ["fortran", "strided", "readonly"])
def test_batch_layouts_preserve_scalar_values(layout):
    x = numpy.random.default_rng(5).normal(size=(2, 3, 500))
    if layout == "fortran":
        x = numpy.asfortranarray(x)
    elif layout == "strided":
        x = x[..., ::-2]
    else:
        x.flags.writeable = False
    actual = sample_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([sample_entropy(row) for row in x.reshape(-1, x.shape[-1])]).reshape(x.shape[:-1])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


def test_batch_default_tolerances_are_per_row():
    x = numpy.random.default_rng(6).normal(size=(4, 400))
    x *= numpy.array([1e-100, 0.01, 1.0, 1e100])[:, None]
    x[0] = 1.0
    actual = sample_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([sample_entropy(row, tolerance=0.2 * numpy.std(row, ddof=0)) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))
    assert numpy.isnan(actual[0])
    numpy.testing.assert_array_equal(sample_entropy(x[:1], tolerance=0.5), [-0.0])


def test_batch_preserves_nan_inf_and_signed_zero():
    x = numpy.array([[0, 1, 2, 3], [0, 0, 1, 2], [1, 1, 1, 1]])
    actual = sample_entropy(x, order=1, tolerance=0.5)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([sample_entropy(row, order=1, tolerance=0.5) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))
    assert numpy.isnan(actual[0]) and numpy.isinf(actual[1])
    assert numpy.signbit(actual[2])


@pytest.mark.parametrize("shape", [(0, 20), (2, 0, 20)])
def test_empty_batch_keeps_leading_shape(shape):
    result = sample_entropy(numpy.empty(shape))
    assert isinstance(result, numpy.ndarray)
    assert result.shape == shape[:-1]
    assert result.dtype == numpy.float64


@pytest.mark.parametrize("value", [numpy.nan, numpy.inf, -numpy.inf])
def test_batch_nonfinite_error_has_full_first_index(value):
    x = numpy.ones((2, 3, 10))
    x[1, 0, 4] = value
    x[1, 2, 0] = value
    with pytest.raises(ValueError, match=r"finite.*index \(1, 0, 4\)"):
        sample_entropy(x)


@pytest.mark.parametrize("x", [1.0, numpy.empty((2, 0)), numpy.ones((2, 3)), [[1, 2], [3]], [[1j] * 10]])
def test_invalid_batch_input_is_rejected(x):
    with pytest.raises(ValueError):
        sample_entropy(x)


@pytest.mark.parametrize("tolerance", [-1, numpy.nan, numpy.inf, True, "0.2", 1j, 10**400])
def test_batch_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        sample_entropy(numpy.ones((2, 10)), tolerance=tolerance)


def test_batch_has_no_axis_argument():
    kwargs = {"axis": -1}
    with pytest.raises(TypeError, match="axis"):
        sample_entropy(numpy.ones((2, 10)), **kwargs)


@pytest.mark.parametrize("layout", ["windows", "fortran", "reversed"])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_direct_2d_views_match_contiguous_scalar_and_std(monkeypatch, layout, order):
    module = importlib.import_module("complexeva.sample_entropy")
    rng = numpy.random.default_rng(26)
    if layout == "windows":
        x = numpy.lib.stride_tricks.sliding_window_view(rng.normal(size=1000), 513)[::61]
    else:
        x = rng.normal(size=(6, 1026)) * 2**30 + 2**45
        x = numpy.asfortranarray(x[:, :513]) if layout == "fortran" else x[:, ::-2]
    x.flags.writeable = False
    expected_tolerances = numpy.array([0.2 * numpy.std(numpy.ascontiguousarray(row), ddof=0) for row in x])
    actual_tolerances = module._batch_tolerances(x)
    numpy.testing.assert_array_equal(actual_tolerances.view(numpy.uint64), expected_tolerances.view(numpy.uint64))
    kernel = module._match_counts_rows

    def checked(values, direct, *args):
        assert values is x
        assert not args[-1]
        assert numpy.shares_memory(direct, x)
        return kernel(values, direct, *args)

    monkeypatch.setattr(module, "_match_counts_rows", checked)
    expected = numpy.array([sample_entropy(numpy.ascontiguousarray(row), order=order) for row in x])
    actual = sample_entropy(x, order=order)
    assert isinstance(actual, numpy.ndarray)
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("n, dtype", [(65538, numpy.int32), (65539, numpy.int64)])
def test_batch_count_width_uses_maximum_pair_count(monkeypatch, n, dtype):
    module = importlib.import_module("complexeva.sample_entropy")
    kernel = module._match_counts_rows
    expected = (n - 2) * (n - 3) // 2

    def checked(values, direct, order, tolerances, chunks, count_dtype, copy_rows):
        assert count_dtype == dtype
        counts = kernel(values, direct, order, tolerances, chunks, count_dtype, copy_rows)
        numpy.testing.assert_array_equal(counts, numpy.full((2, 2), expected, dtype=dtype))
        return counts

    monkeypatch.setattr(module, "_match_counts_rows", checked)
    numpy.testing.assert_array_equal(sample_entropy(numpy.ones((2, n)), tolerance=0.1), [-0.0, -0.0])


@pytest.mark.parametrize("dtype", [numpy.int32, numpy.int64])
def test_tree_index_width_does_not_narrow_pair_counts(dtype):
    n = 100000
    tree = _build(numpy.ones(n), 3, n - 2, 8, dtype)
    assert all(array.dtype == dtype for array in tree[-3:])
    assert _count(*tree, 0.1) == (n - 2) * (n - 3) // 2
    if dtype == numpy.int32:
        assert _tree_workspace_bytes(n - 2, 3) == sum(array.nbytes for array in tree) + (n - 2) * 4


def test_batch_tree_budget_limits_active_rows_without_changing_thread_pool(monkeypatch):
    module = importlib.import_module("complexeva.sample_entropy")
    kernel = module._match_counts_rows
    threads = module.get_num_threads()
    n = 100000

    def checked(values, direct, order, tolerances, chunks, count_dtype, copy_rows):
        expected = min(16, threads, max(1, module._TREE_BATCH_BYTES // _tree_workspace_bytes(n - 2, 3)))
        assert chunks == expected
        return kernel(values, direct, order, tolerances, chunks, count_dtype, copy_rows)

    monkeypatch.setattr(module, "_match_counts_rows", checked)
    numpy.testing.assert_array_equal(sample_entropy(numpy.ones((16, n)), tolerance=0.1), numpy.zeros(16))
    assert module.get_num_threads() == threads
