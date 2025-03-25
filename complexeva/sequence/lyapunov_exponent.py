import numpy as np
from scipy.spatial.distance import pdist, squareform


def lyapunov_exponent(data, tau=1, m=2, dt=1):
    """
    Calculate the largest Lyapunov exponent for a time series.

    Args:
        data: numpy array or list of numerical values
        tau: time delay (default=1)
        m: embedding dimension (default=2)
        dt: time step (default=1)

    Returns:
        float: largest Lyapunov exponent
    """
    x = np.array(data)
    N = len(x)
    Y = np.zeros((N - (m - 1) * tau, m))
    for i in range(m):
        Y[:, i] = x[i * tau : N - (m - 1) * tau + i * tau]
    D = squareform(pdist(Y))
    np.fill_diagonal(D, np.inf)  # Exclude self-matches
    nearest = np.argmin(D, axis=1)
    div = np.zeros(N - (m - 1) * tau - 1)
    for i in range(len(div)):
        div[i] = np.abs(x[i + 1] - x[nearest[i] + 1])
    positive_div = div[div > 0]  # Avoid log(0)
    if len(positive_div) == 0:
        return 0.0
    return -np.mean(np.log(positive_div)) / dt
