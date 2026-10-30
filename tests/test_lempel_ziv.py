import re

import numpy
import pytest

from complexeva import lempel_ziv, lempel_ziv_hilbert


def reference_kaspar_schuster(s):
    """Kaspar & Schuster (1987) algorithm as transcribed on Wikipedia, for strings of length >= 2."""
    i, k, start, c, k_max, n = 0, 1, 1, 1, 1, len(s)
    while True:
        if s[i + k - 1] == s[start + k - 1]:
            k += 1
            if start + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == start:
                c += 1
                start += k_max
                if start + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


@pytest.mark.parametrize(
    "s, expected",
    [("1", 1), ("0000000000", 2), ("0101010101", 3), ("0001101001000101", 6), ("1001111011000010", 6)],
)
def test_known_strings(s, expected):
    assert lempel_ziv(s) == expected


def test_matches_reference_on_random_strings():
    rng = numpy.random.default_rng(0)
    for _ in range(3000):
        s = "".join(rng.choice(["0", "1"], size=int(rng.integers(2, 300))))
        assert lempel_ziv(s) == reference_kaspar_schuster(s), s


def test_matches_reference_on_every_short_word():
    for n in range(2, 13):
        for word in range(1 << n):
            s = format(word, f"0{n}b")
            assert lempel_ziv(s) == reference_kaspar_schuster(s), s


def test_matches_reference_on_biased_periodic_and_repeated_strings():
    rng = numpy.random.default_rng(7)
    half = rng.integers(0, 2, 700)
    cases = [rng.random(1500) < p for p in (0.01, 0.1, 0.9, 0.99)]
    cases += [numpy.resize(rng.integers(0, 2, period), 1500) for period in (1, 2, 7, 64)]
    cases += [numpy.concatenate([half, half]), numpy.concatenate([half, half[::-1], half])]
    for bits in cases:
        s = "".join("1" if bit else "0" for bit in bits)
        assert lempel_ziv(s) == reference_kaspar_schuster(s), s[:40]


@pytest.mark.parametrize("method", ["median", "mean", "diff"])
def test_numeric_input_equals_binarised_string(method):
    x = numpy.random.default_rng(1).standard_normal(400)
    if method == "diff":
        bits = numpy.diff(x) > 0
    else:
        bits = x > (numpy.median(x) if method == "median" else numpy.mean(x))
    assert lempel_ziv(x, method=method) == reference_kaspar_schuster("".join("1" if b else "0" for b in bits))


def test_normalize_default_off_and_formula():
    s = "1001111011000010"
    assert lempel_ziv(s) == 6 and isinstance(lempel_ziv(s), int)
    assert lempel_ziv(s, normalize=True) == pytest.approx(6 * numpy.log2(16) / 16)
    assert lempel_ziv("1", normalize=True) == 0.0


def test_normalized_random_near_one_and_regular_low():
    rng = numpy.random.default_rng(4)
    assert lempel_ziv(rng.standard_normal(20000), normalize=True) == pytest.approx(1.0, abs=0.1)
    assert lempel_ziv(numpy.sin(numpy.linspace(0, 40 * numpy.pi, 20000)), normalize=True) < 0.05


def test_normalize_uses_binary_length_for_diff():
    x = numpy.random.default_rng(5).standard_normal(300)
    raw = lempel_ziv(x, method="diff")
    assert lempel_ziv(x, method="diff", normalize=True) == pytest.approx(raw * numpy.log2(299) / 299)


def test_hilbert_variant_forwards_normalize():
    x = numpy.random.default_rng(6).standard_normal(500)
    assert 0 < lempel_ziv_hilbert(x, normalize=True) < 2


def test_hilbert_variant_runs():
    assert lempel_ziv_hilbert(numpy.random.default_rng(2).standard_normal(500)) > 1


@pytest.mark.parametrize(
    "measure, data",
    [(lempel_ziv, numpy.array(1.0)), (lempel_ziv_hilbert, numpy.array(1.0)), (lempel_ziv_hilbert, numpy.ones((2, 10)))],
)
def test_invalid_shape_names_bad_shape(measure, data):
    with pytest.raises(ValueError, match=re.escape(str(data.shape))):
        measure(data)


@pytest.mark.parametrize("measure", [lempel_ziv, lempel_ziv_hilbert])
def test_invalid_method_names_bad_value(measure):
    with pytest.raises(ValueError, match="method.*'mode'"):
        measure(numpy.arange(100.0), method="mode")


@pytest.mark.parametrize("data", ["0102", "not binary"])
def test_nonbinary_string_names_bad_value(data):
    with pytest.raises(ValueError) as error:
        lempel_ziv(data)
    assert "time_series" in str(error.value)
    assert repr(data) in str(error.value)


@pytest.mark.parametrize("measure", [lempel_ziv, lempel_ziv_hilbert])
@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.int16])
def test_inputs_computed_in_float64(measure, dtype):
    x = (numpy.random.default_rng(8).standard_normal(100) * 100).astype(dtype)
    assert measure(x) == measure(x.astype(numpy.float64))


def test_binary_string_ignores_method():
    assert lempel_ziv("0101", method="unused") == lempel_ziv("0101")


@pytest.mark.parametrize("method", ["median", "mean", "diff"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("shape", [(4, 0), (4, 1), (4, 2), (2, 3, 261), (0, 8), (2, 0, 8)])
def test_numeric_batch_bitwise_equals_1d(method, normalize, shape):
    x = numpy.random.default_rng(20).standard_normal(shape)
    if x.size:
        x[0] = 0
        if shape[0] > 1:
            x[1] += 1000
    actual = lempel_ziv(x, method=method, normalize=normalize)
    assert isinstance(actual, numpy.ndarray)
    dtype = numpy.float64 if normalize else numpy.int64
    expected = numpy.array(
        [lempel_ziv(x[index], method=method, normalize=normalize) for index in numpy.ndindex(shape[:-1])], dtype=dtype
    )
    assert actual.shape == shape[:-1]
    assert actual.dtype == dtype
    numpy.testing.assert_array_equal(actual.ravel().view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("method", ["median", "mean", "diff"])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly", "unaligned"])
def test_batch_layout_preserves_1d_bits(method, layout):
    x = numpy.random.default_rng(21).standard_normal((2, 3, 1000))
    if layout == "F":
        x = numpy.asfortranarray(x)
    elif layout == "strided":
        x = x[..., ::-2]
    elif layout == "readonly":
        x.flags.writeable = False
    elif layout == "unaligned":
        unaligned = numpy.ndarray(x.shape, dtype=numpy.float64, buffer=bytearray(x.nbytes + 1), offset=1)
        unaligned[:] = x
        x = unaligned
    actual = lempel_ziv(x, method=method, normalize=True)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([lempel_ziv(x[index], method=method, normalize=True) for index in numpy.ndindex(x.shape[:-1])])
    numpy.testing.assert_array_equal(actual.ravel().view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize(
    "dtype", [numpy.bool_, numpy.int16, numpy.uint64, numpy.float16, numpy.float32, numpy.float64, numpy.longdouble, ">f8"]
)
@pytest.mark.parametrize("method", ["median", "mean", "diff"])
def test_batch_dtypes_match_1d_float64_conversion(dtype, method):
    x = numpy.asfortranarray(numpy.random.default_rng(22).uniform(0, 10, size=(2, 3, 32)).astype(dtype))
    actual = lempel_ziv(x, method=method, normalize=True)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([lempel_ziv(x[index], method=method, normalize=True) for index in numpy.ndindex(x.shape[:-1])])
    numpy.testing.assert_array_equal(actual.ravel().view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("method", ["median", "mean", "diff"])
def test_strided_window_batch_equals_each_window(method):
    series = numpy.random.default_rng(23).standard_normal(4000)
    x = numpy.lib.stride_tricks.sliding_window_view(series, 1000)[::100]
    actual = lempel_ziv(x, method=method, normalize=True)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([lempel_ziv(row, method=method, normalize=True) for row in x])
    numpy.testing.assert_array_equal(actual.view(numpy.uint64), expected.view(numpy.uint64))


@pytest.mark.parametrize("value", [numpy.nan, numpy.inf, -numpy.inf])
@pytest.mark.parametrize("method", ["median", "mean", "diff"])
def test_batch_nonfinite_names_full_index(value, method):
    x = numpy.asfortranarray(numpy.ones((2, 3, 8)))
    x[0, 1, 5] = value
    x[1, 0, 0] = numpy.nan
    with pytest.raises(ValueError, match="time_series.*finite") as error:
        lempel_ziv(x, method=method)
    assert str(value) in str(error.value)
    assert "index (0, 1, 5)" in str(error.value)


@pytest.mark.parametrize("data", [numpy.ones((2, 3), dtype=complex), [["0", "1"], ["1", "0"]], [[1], [2, 3]]])
def test_invalid_numeric_batch(data):
    with pytest.raises(ValueError, match="time_series"):
        lempel_ziv(data)


def test_batch_unknown_method_names_bad_value():
    with pytest.raises(ValueError, match="method.*'mode'"):
        lempel_ziv(numpy.ones((2, 8)), method="mode")


@pytest.mark.parametrize("n", [63, 64, 65, 511, 512, 513, 8191, 8192, 8193])
def test_packed_word_and_prefix_index_boundaries(n):
    bits = numpy.random.default_rng(n).integers(0, 2, n)
    word = "".join("1" if bit else "0" for bit in bits)
    assert lempel_ziv(word) == reference_kaspar_schuster(word)


@pytest.mark.parametrize("gap", [65534, 65535, 65536, 70000])
def test_compact_occurrence_distances_do_not_wrap(gap):
    bits = numpy.zeros(gap + 300, dtype=numpy.bool_)
    bits[[0, 1, 4, gap, gap + 1, gap + 5, gap + 100, gap + 104]] = True
    # Exact suffix-automaton fixture: the uint16 escape must not discard later occurrences.
    word = "".join("1" if bit else "0" for bit in bits)
    assert lempel_ziv(word) == 8
    actual = lempel_ziv(numpy.array([bits], dtype=numpy.float64), method="mean")
    assert isinstance(actual, numpy.ndarray)
    assert actual[0] == 8


@pytest.mark.parametrize("n", [1, 2, 3, 4, 9, 64, 261, 1000])
def test_batch_median_selection_matches_numpy_with_ties_and_extremes(n):
    x = numpy.random.default_rng(n).integers(-3, 4, size=(2, 3, n)).astype(numpy.float64)
    x[0, 0] = numpy.finfo(numpy.float64).max
    x[0, 1] = -numpy.finfo(numpy.float64).max
    x[1, 0] = numpy.resize([-0.0, 0.0, numpy.nextafter(0.0, 1.0)], n)
    x = x.transpose(1, 0, 2)[..., ::-1]
    x.flags.writeable = False
    original = x.copy()
    with numpy.errstate(over="ignore", invalid="ignore"):
        actual = lempel_ziv(x)
        assert isinstance(actual, numpy.ndarray)
        expected = numpy.array([lempel_ziv(x[index]) for index in numpy.ndindex(x.shape[:-1])])
    numpy.testing.assert_array_equal(actual.ravel(), expected)
    numpy.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32, numpy.uint64])
@pytest.mark.parametrize("layout", ["C", "windows", "transposed_reversed"])
def test_batch_reads_original_layout_without_float64_row_buffers(monkeypatch, dtype, layout):
    import importlib

    module = importlib.import_module("complexeva.lempel_ziv")
    x = numpy.random.default_rng(24).integers(0, 16, size=(3, 2, 256)).astype(dtype)
    if layout == "windows":
        x = numpy.lib.stride_tricks.sliding_window_view(x[0, 0], 128)[::17]
    elif layout == "transposed_reversed":
        x = x.transpose(1, 0, 2)[..., ::-2]
    x.flags.writeable = False
    original = x.copy()
    kernel = module._lz_chunk

    def checked(values, *args):
        assert values is x
        return kernel(values, *args)

    monkeypatch.setattr(module, "_lz_chunk", checked)
    actual = lempel_ziv(x)
    assert isinstance(actual, numpy.ndarray)
    expected = numpy.array([lempel_ziv(x[index]) for index in numpy.ndindex(x.shape[:-1])])
    numpy.testing.assert_array_equal(actual.ravel(), expected)
    numpy.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("method", ["median", "mean", "diff"])
@pytest.mark.parametrize("invalid", [numpy.nan, numpy.inf, -numpy.inf])
@pytest.mark.filterwarnings("error")
def test_nonfinite_order_with_transposed_negative_strides(method, invalid):
    x = numpy.ones((3, 2, 256)).transpose(1, 0, 2)[..., ::-2]
    x[0, 2, 99] = invalid
    x[1, 0, 0] = invalid
    with pytest.raises(ValueError, match=r"finite.*index \(0, 2, 99\)"):
        lempel_ziv(x, method=method)
