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
