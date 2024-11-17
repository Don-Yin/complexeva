"""
    Fractal adjacency matrix generation and analysis with varying degrees of complexity
"""

import numpy as np
import matplotlib.pyplot as plt


class DummyAdjacencyMatrix:
    def __init__(self, size, fractal_dimension):
        """Initialize the FractalAdjacencyMatrix with a given size and fractal dimension.

        Parameters:
        size (int): The desired size of the adjacency matrix.
        fractal_dimension (float): The target fractal dimension.
        """
        self.size, self.fractal_dimension = size, fractal_dimension

    def generate_matrix(self, remove_upper_triangle=False, save_as=None):
        """Generate the fractal adjacency matrix based on the specified fractal dimension.

        Parameters:
        fractal_dimension (float): The target fractal dimension.

        Returns:
        numpy.ndarray: The generated adjacency matrix.
        """
        max_m = min(self.size, 10)
        m, pattern = self.calculate_m_and_pattern(self.fractal_dimension, max_m)
        iterations = int(np.ceil(np.log(self.size) / np.log(m)))
        actual_size = m**iterations
        matrix = np.ones((actual_size, actual_size), dtype=bool)
        self.apply_pattern(matrix, 0, 0, actual_size, m, pattern, iterations)

        # Resize matrix if needed
        if actual_size != self.size:
            if actual_size > self.size:
                matrix = matrix[: self.size, : self.size]
            else:
                pad_size = self.size - actual_size
                matrix = np.pad(matrix, ((0, pad_size), (0, pad_size)), mode="constant", constant_values=False)

        matrix = self.post_processing_matrix(matrix, remove_upper_triangle)

        if save_as:
            save_as = save_as if save_as.endswith(".png") else f"{save_as}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(matrix, cmap="binary", interpolation="nearest")
            plt.title("Dummy Fractal Adjacency Matrix / FD ~ {:.2f}".format(self.fractal_dimension))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_as, dpi=420, bbox_inches="tight")
            plt.close()

        return matrix

    @staticmethod
    def post_processing_matrix(matrix, remove_upper_triangle=False):
        matrix = matrix.astype(int)
        if remove_upper_triangle:
            matrix = np.tril(matrix)
        return matrix

    def calculate_m_and_pattern(self, target_dimension, max_scaling_factor=10):
        """Calculate the scaling factor 'm' and the base pattern for the fractal.

        Parameters:
        target_dimension (float): The target fractal dimension.
        max_m (int): The maximum value of 'm' to consider (default is 10).

        Returns:
        tuple: A tuple containing 'm' and the 'pattern' numpy.ndarray.
        """
        best_dimension_diff = float("inf")
        best_scaling_factor = None
        best_pattern = None

        for scaling_factor in range(2, max_scaling_factor + 1):
            total_cells = scaling_factor * scaling_factor
            # Iterate over possible numbers of cells to remove
            for removed_cells in range(1, total_cells):
                remaining_cells = total_cells - removed_cells
                fractal_dim = np.log(remaining_cells) / np.log(scaling_factor)
                dimension_diff = abs(fractal_dim - target_dimension)

                if dimension_diff < best_dimension_diff:
                    best_dimension_diff = dimension_diff
                    best_scaling_factor = scaling_factor
                    # Create the base pattern
                    pattern = np.ones((scaling_factor, scaling_factor), dtype=bool)
                    # Determine which cells to remove, starting from the center
                    indices = [(i, j) for i in range(scaling_factor) for j in range(scaling_factor)]
                    center = scaling_factor // 2
                    indices_sorted = sorted(
                        indices,
                        key=lambda x: (abs(x[0] - center) + abs(x[1] - center), x[0], x[1]),
                    )
                    for i, j in indices_sorted[:removed_cells]:
                        pattern[i, j] = False
                    best_pattern = pattern.copy()

                if best_dimension_diff == 0:
                    return best_scaling_factor, best_pattern

        return best_scaling_factor, best_pattern

    def apply_pattern(self, matrix, x_start, y_start, size, m, pattern, iterations_left):
        """Recursively apply the fractal pattern to the matrix.

        Parameters:
        matrix (numpy.ndarray): The matrix to modify.
        x_start (int): The starting x-coordinate in the matrix.
        y_start (int): The starting y-coordinate in the matrix.
        size (int): The size of the current square in the matrix.
        m (int): The scaling factor.
        pattern (numpy.ndarray): The base pattern to apply.
        iterations_left (int): The number of iterations left to apply.
        """
        if iterations_left == 0:
            return
        step = size // m
        for i, j in np.ndindex(m, m):
            x_slice = slice(x_start + j * step, x_start + (j + 1) * step)
            y_slice = slice(y_start + i * step, y_start + (i + 1) * step)
            if not pattern[i, j]:
                matrix[y_slice, x_slice] = False
            else:
                self.apply_pattern(
                    matrix,
                    x_start + j * step,
                    y_start + i * step,
                    step,
                    m,
                    pattern,
                    iterations_left - 1,
                )


if __name__ == "__main__":
    size = 256
    fractal_dimensions = np.linspace(1.0, 2.0, 10)

    for idx, dimension in enumerate(fractal_dimensions):
        generator = DummyAdjacencyMatrix(size, dimension)
        plt.figure(figsize=(6, 6))
        plt.title(f"Fractal Dimension ~ {dimension:.2f}", pad=20)
        plt.imshow(generator.matrix, cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.savefig(f"fractal_{idx}.png")
        plt.close()
