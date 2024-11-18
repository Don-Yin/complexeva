#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

"""
- validating measures of complexity is actually working using synthetic data on various Hurst exponents
"""

import numpy
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from donware import inspect_package
from complexeva.measures.dimension_1.higuchi import hfd_pyeeg, hfd_matlab_equivalent
from complexeva.measures.dimension_1.lempel_ziv import lempel_ziv_v1, lempel_ziv_v2, lempel_ziv, lempel_ziv_v4


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

    def plot_hfd_vs_hurst(self):
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

        # Original HFD vs Hurst plot
        plt.figure(figsize=(9, 7))
        plt.plot(hurst_exponent, expected_hfd, label="Expected HFD")
        for i, func in enumerate(self.funcs):
            plt.plot(hurst_exponent, hfd_results[i], label=f"HFD ({func.__name__})")
        plt.xlabel("Hurst Exponent (H)", fontsize=14)
        plt.ylabel("Higuchi Fractal Dimension (HFD)", fontsize=14)
        plt.title(f"HFD vs Hurst Exponent ({len(hurst_exponent)} points)", fontsize=16)
        plt.tight_layout()
        plt.grid(True)
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=21)
        plt.legend(fontsize=21, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.savefig("hfd_vs_hurst.png", dpi=320, bbox_inches="tight")


if __name__ == "__main__":
    validator = ComplexityMeasuresValidator2D([hfd_pyeeg, hfd_matlab_equivalent])
    validator.plot_hfd_vs_hurst()

    validator = ComplexityMeasuresValidator2D([lempel_ziv_v1, lempel_ziv_v2, lempel_ziv, lempel_ziv_v4])
    validator.plot_hfd_vs_hurst()
