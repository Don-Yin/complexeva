"""Draw the small schematic for each measure in the README table; requires matplotlib.

Each icon is computed from a short synthetic signal, so it shows what the measure looks at.
"""

import argparse
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WIDTH_CM, HEIGHT_CM = 3.6, 1.6
GREY, BLUE, ORANGE = "#9ca3af", "#2563eb", "#ea580c"


def _canvas(columns=1, ratios=None):
    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54),
        gridspec_kw={"width_ratios": ratios} if ratios else None,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96, wspace=0.12)
    for ax in np.atleast_1d(axes):
        ax.set_axis_off()
        ax.margins(0.04, 0.12)
    return fig, axes


def _signal(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def hfd():
    """Signal with a coarser curve that keeps every k-th sample."""
    x = _signal()
    fig, ax = _canvas()
    ax.plot(x, color=GREY, lw=1.0)
    ax.plot(np.arange(0, len(x), 6), x[::6], "-o", color=BLUE, lw=1.4, ms=2.5)
    return fig


def lempel_ziv():
    """Median-binarised samples, split into Kaspar-Schuster components."""
    x = _signal(24, seed=4)
    bits = (x > np.median(x)).astype(int)
    fig, ax = _canvas()
    ax.plot(x, color=GREY, lw=1.0)
    low, high = x.min(), x.max()
    cell = (high - low) * 0.28
    base = low - cell * 1.6
    ax.bar(np.arange(len(bits)), cell, bottom=base, width=1.0, color=np.where(bits, "#111827", "white"), edgecolor=GREY, lw=0.4)
    for boundary in _components(bits):
        ax.plot([boundary - 0.5] * 2, [base - cell * 0.3, base + cell * 1.3], color=BLUE, lw=1.4)
    return fig


def _components(bits):
    """Start positions of each new component after the first symbol (Lempel-Ziv 1976 parsing)."""
    text = "".join(map(str, bits))
    starts, start = [], 1
    for end in range(2, len(text) + 1):
        if text[start:end] not in text[: end - 1]:
            starts.append(start)
            start = end
    return starts + [start] if start < len(text) else starts


def sample_entropy():
    """A template and the later stretches that stay within its tolerance band."""
    t = np.arange(60)
    x = np.sin(t / 3.0) + 0.08 * np.random.default_rng(2).standard_normal(60)
    order, r, i = 3, 0.3, 4
    fig, ax = _canvas()
    ax.plot(x, color=GREY, lw=1.0)
    template = x[i : i + order]
    for j in range(len(x) - order + 1):
        if j != i and np.max(np.abs(x[j : j + order] - template)) < r:
            ax.plot(t[j : j + order], x[j : j + order], color=BLUE, lw=2.2)
    for edge in (template - r, template + r):
        ax.plot(t[i : i + order], edge, color=ORANGE, lw=0.8, ls=":")
    ax.plot(t[i : i + order], template, color=ORANGE, lw=2.2)
    return fig


def permutation_entropy():
    """Histogram of the ordinal patterns of order 3."""
    x = _signal(80, seed=3)
    patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(x, 3), axis=1, kind="stable")
    codes = patterns @ np.array([9, 3, 1])
    counts = [
        np.sum(codes == c)
        for c in sorted({int(p @ [9, 3, 1]) for p in np.array(list(np.ndindex(3, 3, 3))) if len(set(p)) == 3})
    ]
    fig, (left, right) = _canvas(2, [1.2, 1])
    left.plot(x[:30], color=GREY, lw=1.0)
    left.plot(np.arange(8, 11), x[8:11], "-o", color=BLUE, lw=1.4, ms=2.5)
    right.bar(range(6), counts, width=0.75, color=BLUE)
    return fig


def dfa():
    """Cumulative sum with a straight-line trend fitted in each window."""
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(64)
    y = np.cumsum(noise - noise.mean())
    fig, ax = _canvas()
    ax.plot(y, color=GREY, lw=1.0)
    for start in range(0, 64, 16):
        t = np.arange(start, start + 16)
        ax.plot(t, np.polyval(np.polyfit(t, y[t], 1), t), color=ORANGE, lw=1.6)
        ax.axvline(start - 0.5, color=GREY, lw=0.5, ls=":")
    return fig


def lyapunov_exponent():
    """Two nearby starts of the logistic map drifting apart."""
    a, b = np.empty(24), np.empty(24)
    a[0], b[0] = 0.3, 0.3001
    for k in range(1, 24):
        a[k], b[k] = 3.9 * a[k - 1] * (1 - a[k - 1]), 3.9 * b[k - 1] * (1 - b[k - 1])
    fig, ax = _canvas()
    ax.plot(a, color=BLUE, lw=1.2)
    ax.plot(b, color=ORANGE, lw=1.2, ls="--")
    return fig


def compression_complexity():
    """Raw and zlib-compressed byte counts of a regular and a noisy array."""
    rng = np.random.default_rng(6)
    regular = (100 + 60 * np.sin(np.arange(4096) / 9) + rng.integers(0, 4, 4096)).astype(np.uint8)
    noisy = rng.integers(0, 200, 4096).astype(np.uint8)
    fig, ax = _canvas()
    for row, data, colour in ((1, regular, BLUE), (0, noisy, ORANGE)):
        ax.barh(row, 1.0, height=0.6, color="white", edgecolor=GREY, lw=0.6)
        ax.barh(row, len(zlib.compress(data.tobytes())) / data.nbytes, height=0.6, color=colour)
    ax.set_xlim(0, 1.05)
    return fig


ICONS = {
    f.__name__: f
    for f in (hfd, lempel_ziv, sample_entropy, permutation_entropy, dfa, lyapunov_exponent, compression_complexity)
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "figures" / "icons")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, build in ICONS.items():
        fig = build()
        fig.savefig(output_dir / f"{name}.png", dpi=200)
        plt.close(fig)
        print(output_dir / f"{name}.png")


if __name__ == "__main__":
    main()
