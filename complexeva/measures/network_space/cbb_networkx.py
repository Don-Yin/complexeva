#!/Users/donyin/miniconda3/bin/python
import random
from PIL import Image
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from networkx.exception import NetworkXError
import matplotlib.animation as animation


"""
reference:
- https://github.com/PeterTKovacs/boxes/blob/main/boxes/cbb.py
"""


class FractalHandlerNetworkX:
    """
    Handles fractal analysis of NetworkX graphs using box-covering method.
    """

    def __init__(self, network: nx.Graph):
        self.network = network

    def auto_pick_box_sizes(self):
        diameter = nx.diameter(self.network) if nx.is_connected(self.network) else 1
        min_size = max(1, diameter // 21)
        max_size = diameter
        return np.logspace(np.log10(min_size), np.log10(max_size), num=12, dtype=int)

    def compute_fractal_dimensions(self, box_sizes="auto", iterations=10, save_plot_as=None, save_animated_as=None):
        box_sizes = self.auto_pick_box_sizes() if box_sizes == "auto" else box_sizes
        network_image, box_counts = self.plot_network(), []

        # If animation is requested, set up storage for animation frames
        if save_animated_as:
            animation_data = []
            pos = nx.spring_layout(self.network, dim=3)  # 3D layout

            # Create a color mapping based on z-coordinates
            z_coords = np.array([pos[node][2] for node in self.network.nodes()])
            z_min, z_max = z_coords.min(), z_coords.max()
            node_colors = {}
            for node in self.network.nodes():
                z = pos[node][2]
                # Normalize z to [0,1] range for color mapping
                color_val = (z - z_min) / (z_max - z_min) if z_max != z_min else 0
                node_colors[node] = plt.cm.rainbow(color_val)

        for max_distance in tqdm(box_sizes, desc="Box sizes"):
            num_boxes_list = []
            boxes_for_animation = []
            for _ in tqdm(range(iterations), desc=f"Networks for size {max_distance}", leave=False):
                boxes = self.random_box_covering(max_distance)
                num_boxes_list.append(len(boxes))
                if save_animated_as:
                    boxes_for_animation.append(boxes)
            avg_num_boxes = sum(num_boxes_list) / iterations
            box_counts.append(avg_num_boxes)

            if save_animated_as:
                # For animation, store the boxes from the first iteration
                animation_data.append(
                    {
                        "boxes": boxes_for_animation[0],
                        "max_distance": max_distance,
                        "num_boxes": num_boxes_list[0],
                    }
                )

        log_box_sizes, log_counts = np.log(box_sizes), np.log(box_counts)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_box_sizes, log_counts)
        except ValueError:
            slope, intercept, r_value = 0, 0, 0
        fractal_dimension = -slope

        if save_plot_as:
            save_plot_as = save_plot_as if save_plot_as.endswith(".png") else f"{save_plot_as}.png"
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(log_box_sizes, log_counts, color="blue", label="Data")
            ax.plot(
                log_box_sizes,
                slope * log_box_sizes + intercept,
                color="red",
                label=f"Fit (D={fractal_dimension:.2f})",
                linestyle="-",
            )
            ax.set_xlabel("log(Box Size)", fontsize=21)
            ax.set_ylabel("log(Number of Boxes)", fontsize=21)
            ax.set_title("Fractal Dimension Regression", fontsize=21)
            ax.legend(fontsize=21)
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=21)
            fig.canvas.draw()
            plot_image = np.array(fig.canvas.renderer.buffer_rgba())
            network_height, network_width = network_image.shape[:2]
            plot_height, plot_width = plot_image.shape[:2]
            aspect_ratio = network_width / network_height
            new_network_height = plot_height
            new_network_width = int(new_network_height * aspect_ratio)
            resized_network_image = np.array(Image.fromarray(network_image).resize((new_network_width, new_network_height)))
            resized_network_image = resized_network_image[:, :, :3]
            plot_image = plot_image[:, :, :3]
            combined_image = np.hstack((resized_network_image, plot_image))
            plt.figure(
                figsize=(
                    combined_image.shape[1] / 100,
                    combined_image.shape[0] / 100,
                )
            )
            plt.imshow(combined_image)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_plot_as, dpi=420, bbox_inches="tight")
            plt.close()

        # Create and save animation if requested
        if save_animated_as:
            fig = plt.figure(figsize=(15, 8))  # Wider figure to accommodate legend
            gs = plt.GridSpec(1, 2, width_ratios=[2, 1])  # 2:1 ratio for plot and legend
            ax = fig.add_subplot(gs[0], projection="3d")
            legend_ax = fig.add_subplot(gs[1])  # Create legend subplot once
            legend_ax.axis("off")
            legend_ax.spines["top"].set_visible(False)

            def update(frame):
                ax.clear()
                legend_ax.clear()  # Clear previous legend

                data = animation_data[frame]
                boxes = data["boxes"]
                max_distance = data["max_distance"]
                num_boxes = data["num_boxes"]

                # Draw network and boxes in 3D subplot
                edge_list = list(self.network.edges())

                # Assign box colors based on average z-coordinate of nodes in each box
                box_colors = []
                for box in boxes:
                    box_nodes = list(box)
                    box_colors.append(np.mean([node_colors[node] for node in box_nodes], axis=0))

                # Create a mapping of nodes to their box colors
                node_to_color = {}
                for box, color in zip(boxes, box_colors):
                    for node in box:
                        node_to_color[node] = color

                # Draw edges with box colors
                for edge in edge_list:
                    start = pos[edge[0]]
                    end = pos[edge[1]]
                    # Use color of both nodes for edge
                    start_color = node_to_color.get(edge[0], "gray")
                    end_color = node_to_color.get(edge[1], "gray")
                    edge_color = start_color if np.array_equal(start_color, end_color) else "gray"

                    mid_point = (start + end) / 2
                    offset = np.random.normal(0, 0.05, 3)
                    control_point = mid_point + offset
                    t = np.linspace(0, 1, 20)
                    curve = np.array([(1 - t) ** 2 * start + 2 * (1 - t) * t * control_point + t**2 * end for t in t])
                    ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2], c=edge_color, alpha=0.7, linewidth=1.5)

                # Draw boxes and create legend elements
                legend_elements = []
                for box_idx, (box, color) in enumerate(zip(boxes, box_colors)):
                    box_nodes = list(box)
                    box_pos = np.array([pos[node] for node in box_nodes])
                    scatter = ax.scatter(
                        box_pos[:, 0], box_pos[:, 1], box_pos[:, 2], c=[color], s=150, alpha=0.8, edgecolor="white"
                    )
                    legend_elements.append((scatter, f"Box {box_idx+1}"))

                # Set 3D plot properties
                ax.set_title(f"Step {frame+1}/{len(animation_data)}\nBox Size = {max_distance}\nBoxes Required: {num_boxes}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                # Calculate optimal number of columns based on number of boxes
                n_boxes = len(boxes)
                n_cols = min(4, n_boxes)  # Max 4 columns for more spacing
                n_rows = (n_boxes + n_cols - 1) // n_cols  # Ceiling division

                # Create legend with more spacing
                legend = legend_ax.legend(
                    *zip(*legend_elements),
                    loc="center",
                    ncol=n_cols,
                    bbox_to_anchor=(0.5, 0.5),
                    borderaxespad=1,
                    labelspacing=1.5,  # Increase vertical spacing between legend entries
                    handletextpad=1.2,  # Increase spacing between marker and text
                    title="Box Groups",
                    frameon=True,  # Add frame around legend
                    fancybox=True,  # Round corners
                    shadow=True,  # Add shadow
                )
                legend.get_title().set_fontsize(14)
                for text in legend.get_texts():
                    text.set_fontsize(12)

            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=20, azim=45)
            ax.dist = 8

            ani = animation.FuncAnimation(fig, update, frames=len(animation_data), interval=2000, repeat=True)

            save_animated_as = save_animated_as if save_animated_as.endswith((".png", ".gif")) else save_animated_as + ".gif"
            ani.save(save_animated_as.replace(".png", ".gif"), writer="pillow", fps=0.5)
            plt.close(fig)

        return fractal_dimension, r_value**2, log_box_sizes, log_counts

    # ---- functional helpers ----
    def random_box_covering(self, max_distance: int):
        """
        [NOTE] This implementation handles disconnected graphs by processing each connected component separately.

        This function implements the random box covering algorithm to estimate
        the boxes needed to cover a network at a given maximum distance.

        Args:
            max_distance (int): The maximum distance allowed within each box, in number of hops.

        Returns:
            List[Set]: A list of sets, each containing the nodes in a box.
        """
        boxes = []
        for component in nx.connected_components(self.network):
            subgraph = self.network.subgraph(component)
            uncovered_nodes = set(subgraph.nodes())
            while uncovered_nodes:
                p = random.choice(list(uncovered_nodes))
                distances = nx.single_source_shortest_path_length(subgraph, p)
                box_nodes = set(node for node, dist in distances.items() if dist <= max_distance).intersection(uncovered_nodes)
                boxes.append(box_nodes)
                uncovered_nodes -= box_nodes
        return boxes

    def plot_network(self):
        pos = nx.get_node_attributes(self.network, "pos")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if pos:
            # Use provided positions if they exist
            pos_3d = {node: (x, y, 0) for node, (x, y) in pos.items()}  # Convert 2D to 3D
        else:
            # Generate 3D spring layout
            pos_3d = nx.spring_layout(self.network, dim=3)

        # Draw edges with curves
        for edge in self.network.edges():
            start = np.array(pos_3d[edge[0]])
            end = np.array(pos_3d[edge[1]])
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], "gray", alpha=0.5)

        # Draw nodes
        node_pos = np.array([pos_3d[node] for node in self.network.nodes()])
        ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2], c="lightblue", s=30)

        ax.set_title("3D Network Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return image


if __name__ == "__main__":
    from src.dummies.networks import WebFractalNetwork, SierpinskiFractal

    network = SierpinskiFractal(recursion_depth=6)
    network = WebFractalNetwork(complexity=4)

    handler = FractalHandlerNetworkX(network.network)
    handler.compute_fractal_dimensions(box_sizes=np.linspace(1, 10, 20), iterations=10)
