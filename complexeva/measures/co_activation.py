#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

"""
- now we have two 1d series as signals make dummy data using numpy
- use either a sliding window or regression method to produce a new signal that represents the co-activation between the two signals
- in the sliding_window method, the window_size parameter is used to determine the size of the window used to calculate the co-activation
- it moves one step at a time and calculates the correlation between the two signals in the window
- in the regression method, the two signals are used as independent and dependent variables in a linear regression model
- the resulting signal is the predicted dependent variable
- if we use the sliding_window method, the window_size parameter must be an integer greater than 0
- if we use the regression method, we must plot the two signal and the resulting signal
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def get_co_activation(
    signal_1: np.ndarray, signal_2: np.ndarray, method: str, window_size: int = None, save_plot_as=None
) -> np.ndarray:
    """
    Get a measure of co-activation between two 1D signals.

    Args:
        signal_1: First 1D time series signal
        signal_2: Second 1D time series signal
        method: Either 'sliding_window' or 'regression'
        window_size: Size of the window for 'sliding_window' method (must be an integer greater than 0)

    Returns:
        np.ndarray: A new signal representing the co-activation between the two signals
    """
    match method:
        case "sliding_window":
            if window_size is None or window_size <= 0:
                raise ValueError("window_size must be an integer greater than 0 for sliding_window method")
            correlations = []
            num_windows = len(signal_1) - window_size + 1
            for i in range(num_windows):
                window_signal_1 = signal_1[i : i + window_size]
                window_signal_2 = signal_2[i : i + window_size]
                corr = np.corrcoef(window_signal_1, window_signal_2)[0, 1]
                correlations.append(corr)

            # Plot signals and correlations
            correlations = np.array(correlations)
            if save_plot_as:
                plt.figure(figsize=(10, 5))
                plt.plot(signal_1, label="Signal 1", alpha=0.7)
                plt.plot(signal_2, label="Signal 2", alpha=0.7)
                plt.plot(
                    np.arange(window_size // 2, len(correlations) + window_size // 2),
                    correlations,
                    label="Sliding Window Correlation",
                    linestyle="--",
                )
                plt.legend()
                plt.title(f"Sliding Window Method Co-Activation (window size={window_size})")
                plt.savefig(save_plot_as, dpi=320)
                plt.close()

            return correlations

        case "regression":
            # Reshape signals for regression
            X = signal_1.reshape(-1, 1)
            y = signal_2

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)

            # Predict y
            y_pred = model.predict(X)

            # Plot signals
            if save_plot_as:
                plt.figure(figsize=(10, 5))
                plt.plot(signal_1, label="Signal 1 (IV)")
                plt.plot(signal_2, label="Signal 2 (DV)")
                plt.plot(y_pred, label="Predicted DV", linestyle="--")
                plt.legend()
                plt.title("Regression Method Co-Activation")
                plt.savefig(save_plot_as, dpi=320)
                plt.close()

            return y_pred

        case _:
            raise ValueError("method must be 'sliding_window' or 'regression'")


if __name__ == "__main__":
    t = np.linspace(0, 10, 1000)
    signal1 = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(1000)
    signal2 = 0.8 * np.sin(2 * np.pi * 0.5 * t + np.pi / 4) + 0.2 * np.random.randn(1000)

    get_co_activation(signal1, signal2, method="sliding_window", window_size=32, save_plot_as="coactivation_sliding_window.png")
    get_co_activation(signal1, signal2, method="regression", save_plot_as="coactivation_regression.png")
