"""Reproduce the README's synthetic-data checks; requires matplotlib in addition to complexeva."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from complexeva import hfd, lempel_ziv


def fractional_brownian_motion(hurst, n, repeats, rng):
    """Sample fBm paths starting at zero by circulant embedding of their increments.

    Fractional Gaussian noise has covariance
    gamma(k) = (|k - 1|**(2H) - 2|k|**(2H) + |k + 1|**(2H)) / 2.
    A positive circulant embedding gives Gaussian samples with this covariance,
    unlike directly filtering a power-law spectrum, which is only approximate.
    """
    lag = np.arange(n + 1, dtype=float)
    covariance = (np.abs(lag - 1) ** (2 * hurst) - 2 * lag ** (2 * hurst) + (lag + 1) ** (2 * hurst)) / 2
    eigenvalues = np.fft.rfft(np.concatenate((covariance, covariance[-2:0:-1]))).real
    if np.any(eigenvalues < 0):
        raise ValueError("The fGn circulant embedding must have nonnegative eigenvalues.")
    white = rng.standard_normal((repeats, 2 * n))
    increments = np.fft.irfft(np.fft.rfft(white, axis=1) * np.sqrt(eigenvalues), n=2 * n, axis=1)
    return np.column_stack((np.zeros(repeats), np.cumsum(increments[:, : n - 1], axis=1)))


def plot_hfd(output_dir, rng, n, repeats, k_max):
    hursts = np.arange(1, 10) / 10
    estimates = []
    for i, hurst in enumerate(hursts, start=1):
        paths = fractional_brownian_motion(hurst, n, repeats, rng)
        estimates.append([hfd(path, k_max=k_max) for path in paths])
        print(f"[{i}/{len(hursts)}] H={hurst:.1f}: mean HFD={np.mean(estimates[-1]):.3f}, theory={2 - hurst:.1f}", flush=True)
    estimates = np.asarray(estimates)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    ax.plot(hursts, 2 - hursts, "--", color="#374151", label="Theory: D = 2 − H")
    ax.errorbar(
        hursts,
        estimates.mean(axis=1),
        yerr=estimates.std(axis=1, ddof=1),
        fmt="o",
        color="#2563eb",
        capsize=4,
        label=f"HFD: mean ± 1 SD ({repeats} paths)",
    )
    ax.set(
        xlabel="Hurst exponent H",
        ylabel="Higuchi fractal dimension",
        title=f"fBm: estimated versus theoretical dimension\nn = {n:,} samples · k_max = {k_max}",
        xlim=(0.05, 0.95),
        ylim=(1.0, 2.05),
        xticks=hursts,
    )
    ax.legend(loc="upper right", frameon=False)
    path = output_dir / "hfd-fbm.png"
    fig.savefig(path, dpi=140, pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"Saved {path}; largest absolute mean error: {np.max(np.abs(estimates.mean(axis=1) - (2 - hursts))):.4f}", flush=True)


def plot_lempel_ziv(output_dir, rng, n, repeats):
    period = 128
    sine = np.sin(2 * np.pi * np.arange(n) / period)
    noise = rng.standard_normal((repeats, n))
    sine_lz = lempel_ziv(sine, method="median", normalize=True)
    noise_lz = np.array([lempel_ziv(series, method="median", normalize=True) for series in noise])
    means = [sine_lz, noise_lz.mean()]
    colors = ["#d97706", "#2563eb"]

    fig = plt.figure(figsize=(8.4, 4.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1))
    for i, (series, name, color) in enumerate(zip([sine, noise[0]], ["Sine (period 128)", "White noise"], colors)):
        ax = fig.add_subplot(grid[i, 0])
        ax.plot(series[:384], color=color, linewidth=1)
        ax.set(title=name, ylabel="Amplitude", xlim=(0, 383), xticks=[0, 128, 256, 384])
        if i == 1:
            ax.set_xlabel("Sample (first 384 shown)")

    ax = fig.add_subplot(grid[:, 1])
    bars = ax.bar(["Sine", "White noise"], means, yerr=[0, noise_lz.std(ddof=1)], color=colors, width=0.55, capsize=4)
    ax.axhline(1, color="#374151", linestyle="--", linewidth=1, label="Random-sequence limit")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=7)
    ax.set(ylabel="Normalized LZ: c log₂(n) / n", ylim=(0, 1.28))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        f"Lempel–Ziv: periodic versus random input\nn = {n:,} · median binarization · noise: mean ± 1 SD ({repeats} trials)"
    )
    path = output_dir / "lz-sine-noise.png"
    fig.savefig(path, dpi=140, pil_kwargs={"optimize": True})
    plt.close(fig)
    print(
        f"Saved {path}; normalized LZ: sine={sine_lz:.4f}, white noise={noise_lz.mean():.4f} ± {noise_lz.std(ddof=1):.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    rng = np.random.default_rng(0)
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.axisbelow": True,
        }
    ):
        plot_hfd(args.output_dir, rng, n=8192, repeats=32, k_max=10)
        plot_lempel_ziv(args.output_dir, rng, n=8192, repeats=32)


if __name__ == "__main__":
    main()
