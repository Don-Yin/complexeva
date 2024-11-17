import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class FractalHandlerMatrix:
    """
    Handles the fractal dimension calculation of a binary matrix using an improved Compact Box-Burning algorithm.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def compute_fractal_dimension(self, save_plot_as=None, save_animated_as=None):
        """
        Calculate the fractal dimension of the matrix using the improved Compact Box-Burning algorithm.

        Parameters:
            save_plot_as (str, optional): Path to save the static plot showing fractal dimension calculation
            save_animated_as (str, optional): Path to save the animated GIF showing box counting process

        Returns:
            float: The fractal dimension of the matrix.
        """
        assert np.all((self.matrix == 0) | (self.matrix == 1)), "> warning: Matrix should be binary (0s and 1s only)"
        binary_matrix = self.matrix.astype(np.uint8)
        min_dim = min(binary_matrix.shape)
        max_scale = 2 ** int(np.floor(np.log2(min_dim)))

        num_scales = int(np.floor(np.log2(max_scale))) * 10
        scales = np.unique(np.floor(np.logspace(0, np.log2(max_scale), num=num_scales, base=2))).astype(int)
        scales = scales[scales >= 1]

        box_counts = []
        padded_matrices = []

        # Animation setup if needed
        if save_animated_as:
            fig, ax = plt.subplots(figsize=(8, 8))

        for scale in scales:
            padded_size = (
                int(np.ceil(binary_matrix.shape[0] / scale)) * scale,
                int(np.ceil(binary_matrix.shape[1] / scale)) * scale,
            )
            padded_matrix = np.zeros(padded_size, dtype=np.uint8)
            padded_matrix[: binary_matrix.shape[0], : binary_matrix.shape[1]] = binary_matrix

            if save_animated_as:
                padded_matrices.append((padded_matrix, scale))

            reshaped = padded_matrix.reshape(padded_size[0] // scale, scale, padded_size[1] // scale, scale)
            summed = reshaped.sum(axis=(1, 3))
            occupied_boxes = np.count_nonzero(summed)
            box_counts.append(occupied_boxes)

        # calculate fractal dimension
        scales = np.array(scales)
        box_counts = np.array(box_counts)
        nonzero = box_counts > 0
        scales = scales[nonzero]
        box_counts = box_counts[nonzero]

        log_scales = np.log(1 / scales)
        log_box_counts = np.log(box_counts)

        coeffs = np.polyfit(log_scales, log_box_counts, 1)
        fractal_dimension = coeffs[0]

        # save static plot if requested
        if save_plot_as:
            save_plot_as = save_plot_as if save_plot_as.endswith(".png") else save_plot_as + ".png"
            plt.figure(figsize=(8, 6))
            plt.plot(log_scales, log_box_counts, "o-", label="Data")
            plt.plot(
                log_scales,
                np.polyval(coeffs, log_scales),
                "r--",
                label=f"Fit (D = {fractal_dimension:.4f})",
            )
            plt.xlabel("log(1/ε)")
            plt.ylabel("log(N(ε))")
            plt.title("Fractal Dimension via Compact Box-Burning")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_plot_as, dpi=420, bbox_inches="tight")
            plt.close()

        # Create and save animation if requested
        if save_animated_as:

            def init():
                ax.imshow(binary_matrix, cmap="binary", interpolation="nearest")
                ax.axis("off")
                return []

            def update(frame):
                padded_matrix, scale = padded_matrices[frame]
                ax.clear()
                ax.imshow(padded_matrix, cmap="binary", interpolation="nearest")
                ax.set_title(f"Scale: {scale} pixels, Boxes: {box_counts[frame]}")
                ax.axis("off")

                # Overlay red boxes on occupied regions
                for y in range(0, padded_matrix.shape[0], scale):
                    for x in range(0, padded_matrix.shape[1], scale):
                        if np.any(padded_matrix[y : y + scale, x : x + scale]):
                            rect = plt.Rectangle(
                                (x - 0.5, y - 0.5),
                                scale,
                                scale,
                                edgecolor="red",
                                facecolor="none",
                                lw=1,
                            )
                            ax.add_patch(rect)
                return []

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(scales),
                init_func=init,
                blit=False,
                repeat=False,
            )

            if not save_animated_as.endswith((".png", ".gif")):
                save_animated_as += ".gif"
            ani.save(save_animated_as.replace(".png", ".gif"), writer="pillow", fps=2)
            plt.close(fig)

        return fractal_dimension
