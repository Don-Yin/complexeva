#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

"""
- validating measures of complexity is actually working using synthetic data on various Hurst exponents
"""

import numpy
import matplotlib.pyplot as plt
from complexeva.sequence.higuchi import hfd
from complexeva.sequence.lempel_ziv import lempel_ziv


class ComplexityMeasuresValidator2D:
    def __init__(self, funcs, timeseries_len=1000, num_hurst=200):
        self.funcs, self.num_hurst, self.timeseries_len = funcs, num_hurst, timeseries_len

    def fbm(self, hurst):
        """
        Generate fractional Brownian motion (fBm) time series.

        This method uses the spectral synthesis method to generate a fBm time series
        with a specified Hurst exponent.

        Parameters:
        -----------
        hurst : float
            The Hurst exponent, a value between 0 and 1 that characterizes
            the fractal properties of the time series.

        Returns:
        --------
        numpy.ndarray
            A 1D array representing the generated fBm time series.

        Notes:
        ------
        - The length of the generated time series is determined by self.timeseries_len.
        - This method uses the Fourier filtering method to generate fBm.
        - The first frequency component is set to a small non-zero value (1e-6) to avoid division by zero.
        """

        n = self.timeseries_len
        f = numpy.fft.fftfreq(n)
        f[0] = 1e-6  # avoid division by zero
        psd = numpy.abs(f) ** (-2 * hurst - 1)
        gaussian = numpy.random.randn(n) + 1j * numpy.random.randn(n)
        fBm = numpy.fft.ifft(gaussian * numpy.sqrt(psd)).real
        return fBm

    def plot_vs_hurst(self):
        hurst_exponent = numpy.linspace(0.001, 1, self.num_hurst)
        fBm_series = [self.fbm(h) for h in hurst_exponent]
        expected_hfd = [2 - h for h in hurst_exponent]
        hfd_results = []

        for func in self.funcs:
            hfd_results.append([func(series) for series in fBm_series])

        # ---- plotting ----
        fig, axs = plt.subplots(3, 1, figsize=(9, 10))

        axs[0].plot(fBm_series[0])
        axs[0].set_title(f"fBm series with H = {hurst_exponent[0]:.3f}", fontsize=21)
        mid_index = len(hurst_exponent) // 2
        axs[1].plot(fBm_series[mid_index])
        axs[1].set_title(f"fBm series with H = {hurst_exponent[mid_index]:.3f}", fontsize=21)
        axs[2].plot(fBm_series[-1])
        axs[2].set_title(f"fBm series with H = {hurst_exponent[-1]:.3f}", fontsize=21)

        for ax in axs:
            ax.set_xlabel("Time", fontsize=21)
            ax.set_ylabel("Value", fontsize=21)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.grid(True)

        plt.tight_layout()
        plt.savefig("fBm_series_examples.png", dpi=320, bbox_inches="tight")

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Original HFD vs Hurst plot
        ax1.plot(hurst_exponent, expected_hfd, label="Expected FD")
        for i, func in enumerate(self.funcs):
            ax1.plot(hurst_exponent, hfd_results[i], label=f"HFD ({func.__name__})")
        ax1.set_xlabel("Hurst Exponent (H)", fontsize=14)
        ax1.set_ylabel("Raw Complexity", fontsize=14)
        ax1.set_title("Before Normalization", fontsize=16)
        ax1.grid(True)
        ax1.tick_params(axis="both", labelsize=12)
        ax1.legend(fontsize=12)

        # Normalized HFD vs Hurst plot
        ax2.plot(hurst_exponent, expected_hfd, label="Expected FD")
        for i, func in enumerate(self.funcs):
            # Normalize results between 0 and 1
            normalized_results = (hfd_results[i] - numpy.min(hfd_results[i])) / (
                numpy.max(hfd_results[i]) - numpy.min(hfd_results[i])
            )
            ax2.plot(hurst_exponent, normalized_results, label=f"HFD ({func.__name__})")
        ax2.set_xlabel("Hurst Exponent (H)", fontsize=14)
        ax2.set_ylabel("Normalized Complexity", fontsize=14)
        ax2.set_title("After Normalization (0-1)", fontsize=16)
        ax2.grid(True)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig("hfd_vs_hurst_comparison.png", dpi=320, bbox_inches="tight")


if __name__ == "__main__":
    validator = ComplexityMeasuresValidator2D([hfd])
    validator.plot_vs_hurst()

    validator = ComplexityMeasuresValidator2D([lempel_ziv])
    validator.plot_vs_hurst()
