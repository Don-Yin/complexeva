"""Literal-definition and reference-package checks for Chen fuzzy entropy."""

import importlib

import numpy
import pytest

from complexeva.fuzzy_entropy import _PARALLEL_MIN_TEMPLATES, fuzzy_entropy


def literal_entropy(x, order=2, r=None, n=2):
    """Evaluate the centred-template definition with an independent dense calculation."""
    x = numpy.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * numpy.std(x, ddof=0)
    if r == 0:
        return numpy.nan
    sums = []
    for dimension in (order, order + 1):
        templates = numpy.lib.stride_tricks.sliding_window_view(x, dimension)[: len(x) - order].copy()
        templates -= templates.mean(axis=1, keepdims=True)
        distances = numpy.max(numpy.abs(templates[:, None] - templates[None, :]), axis=2)
        membership = numpy.exp(-(distances**n) / r)
        sums.append(numpy.triu(membership, k=1).sum())
    with numpy.errstate(divide="ignore", invalid="ignore"):
        return -numpy.log(sums[1] / sums[0])


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("exponent", [0.5, 1, 2, 3.5])
@pytest.mark.parametrize("radius", [None, 0.01, 0.2, 1.0])
def test_matches_literal_centred_definition(order, exponent, radius):
    x = numpy.random.default_rng(123).normal(size=41)
    assert fuzzy_entropy(x, order=order, r=radius, n=exponent) == pytest.approx(
        literal_entropy(x, order=order, r=radius, n=exponent), abs=2e-13
    )


def test_membership_divides_after_power():
    x = numpy.array([0.0, 1.0, 0.2, 2.0, -0.5, 0.0])
    assert fuzzy_entropy(x, r=0.3) == pytest.approx(literal_entropy(x, r=0.3), abs=1e-14)
    assert abs(fuzzy_entropy(x, r=0.3) - literal_entropy(x, r=0.3**2)) > 0.1


def test_local_means_and_identical_extendible_template_counts():
    x = numpy.array([0.0, 1.0, 1.2, 1.4, 1.6, 9.0])
    assert fuzzy_entropy(x, r=0.3) == pytest.approx(literal_entropy(x, r=0.3), abs=1e-14)
    assert fuzzy_entropy(x + 40.0, r=0.3) == pytest.approx(fuzzy_entropy(x, r=0.3), abs=2e-14)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("exponent", [1, 2, 3.5])
def test_matches_installed_references(order, exponent):
    eh = pytest.importorskip("EntropyHub")
    pe = pytest.importorskip("pyentrp.entropy")
    x = numpy.random.default_rng(2).normal(size=127)
    r = 0.2 * numpy.std(x, ddof=0)
    actual = fuzzy_entropy(x, order=order, r=r, n=exponent)
    assert actual == pytest.approx(eh.FuzzEn(x, m=order, tau=1, r=(r, exponent))[0][-1], abs=5e-13)
    assert actual == pytest.approx(
        pe.fuzzy_entropy(x, sample_length=order, tolerance=r ** (1 / exponent), n=exponent), abs=5e-13
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_neurokit2_agrees_at_exponent_one(order):
    nk = pytest.importorskip("neurokit2")
    x = numpy.random.default_rng(2).normal(size=127)
    r = 0.2 * numpy.std(x, ddof=0)
    assert fuzzy_entropy(x, order=order, r=r, n=1) == pytest.approx(
        nk.entropy_fuzzy(x, dimension=order, delay=1, tolerance=r)[0], abs=5e-13
    )


def test_default_radius_is_population_sd_and_is_per_signal():
    x = numpy.random.default_rng(5).normal(size=(3, 51)) * numpy.array([0.1, 1, 10])[:, None]
    expected = numpy.array([fuzzy_entropy(row, r=0.2 * numpy.std(row, ddof=0)) for row in x])
    actual = fuzzy_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("radius", [None, 0.0])
def test_zero_radius_is_undefined(radius):
    assert numpy.isnan(fuzzy_entropy(numpy.ones(20), r=radius))
    if radius == 0:
        assert numpy.isnan(fuzzy_entropy(numpy.arange(20), r=radius))


@pytest.mark.parametrize("constant", [0.0, 0.1, 3.7, 1e300])
def test_constant_signal_is_undefined_at_default_width(constant):
    x = numpy.full(100, constant)
    assert numpy.isnan(fuzzy_entropy(x))
    batch = fuzzy_entropy(x[None])
    assert isinstance(batch, numpy.ndarray)
    assert numpy.isnan(batch[0])
    assert fuzzy_entropy(x, r=0.2) == 0.0


def test_minimum_length():
    assert fuzzy_entropy([0, 1, 2, 4], r=0.2) == pytest.approx(literal_entropy([0, 1, 2, 4], r=0.2))


def test_matches_without_extensions_give_infinity():
    assert numpy.isinf(fuzzy_entropy([0, 0, 1, 1, 10], r=1e-300))


def test_small_positive_memberships_are_not_rounded_to_zero():
    x = numpy.random.default_rng(123).normal(size=127)
    actual = fuzzy_entropy(x, order=5, r=0.02, n=1)
    assert numpy.isfinite(actual)
    assert actual == pytest.approx(literal_entropy(x, order=5, r=0.02, n=1), abs=1e-13)


def test_complete_numerical_underflow_is_nan_not_division_error():
    x = [0, 1, 5, 100]
    assert numpy.isnan(fuzzy_entropy(x, r=1e-300))


@pytest.mark.parametrize("shape", [(4, 55), (2, 3, 41), (0, 33), (2, 0, 33)])
@pytest.mark.parametrize("radius", [None, 0, 0.2])
def test_batch_shape_dtype_and_bitwise_row_contract(shape, radius):
    x = numpy.random.default_rng(6).normal(size=shape)
    actual = fuzzy_entropy(x, r=radius)
    expected = numpy.array([fuzzy_entropy(row, r=radius) for row in x.reshape(-1, shape[-1])]).reshape(shape[:-1])
    assert isinstance(actual, numpy.ndarray)
    assert actual.shape == shape[:-1]
    assert actual.dtype == numpy.float64
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("layout", ["fortran", "reverse", "strided", "readonly", "float32", "int16", "float16", "bigendian"])
def test_batch_layouts_have_scalar_bits_and_leave_input_unchanged(layout):
    x = numpy.random.default_rng(7).normal(size=(2, 3, 70))
    if layout == "fortran":
        x = numpy.asfortranarray(x)
    elif layout == "reverse":
        x = x[..., ::-1]
    elif layout == "strided":
        x = x[..., ::2]
    elif layout == "readonly":
        x.flags.writeable = False
    elif layout == "bigendian":
        x = x.astype(">f8")
    else:
        x = x.astype(layout)
    before = x.copy()
    actual = fuzzy_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([fuzzy_entropy(x[index]) for index in numpy.ndindex(x.shape[:-1])]).reshape(x.shape[:-1])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))
    numpy.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("count", [1, 2, 13])
@pytest.mark.parametrize("extra", [1, 2, 3])
def test_parallel_and_serial_schedules_use_identical_reduction(count, extra):
    x = numpy.random.default_rng(8).normal(size=(count, _PARALLEL_MIN_TEMPLATES + extra))
    actual = fuzzy_entropy(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([fuzzy_entropy(row) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("value", [numpy.nan, numpy.inf, -numpy.inf])
def test_nonfinite_error_names_full_index(value):
    x = numpy.ones((2, 3, 20))
    x[1, 2, 8] = value
    with pytest.raises(ValueError, match=r"index \(1, 2, 8\)"):
        fuzzy_entropy(x)


@pytest.mark.parametrize("x", [1.0, [], [1], [1, 2], [1, 2, 3], [[1], [2]], [1 + 0j, 2, 3, 4], ["a", "b", "c", "d"]])
def test_invalid_signal_rejected(x):
    with pytest.raises(ValueError):
        fuzzy_entropy(x)


@pytest.mark.parametrize("order", [0, -1, 2.0, True, numpy.nan, "2"])
def test_invalid_order_names_parameter(order):
    with pytest.raises(ValueError, match="order"):
        fuzzy_entropy(numpy.ones(20), order=order)


@pytest.mark.parametrize("parameter", ["r", "n"])
@pytest.mark.parametrize("value", [-1, numpy.nan, numpy.inf, True, "0.2", 1j, 10**400])
def test_invalid_real_parameters_name_parameter(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        fuzzy_entropy(numpy.ones(20), **{parameter: value})


def test_zero_exponent_rejected_and_numpy_scalars_accepted():
    with pytest.raises(ValueError, match="n"):
        fuzzy_entropy(numpy.ones(20), n=0)
    assert fuzzy_entropy(numpy.ones(20), order=numpy.int64(2), r=numpy.float32(0.2), n=numpy.float64(2)) == 0.0


def test_no_fastmath():
    """Production reductions do not allow floating-point reassociation."""
    module = importlib.import_module("complexeva.fuzzy_entropy")
    for name in ("_templates", "_anchor_sums", "_entropy_serial", "_entropy_parallel", "_entropy_rows"):
        assert getattr(module, name).targetoptions.get("fastmath", False) is False
