import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rich import print


def matrix_to_networkx(matrix, method: str = "laplacian", save_as=None):
    """
    Convert a symmetric adjacency matrix to a NetworkX graph.
    Parameters: matrix (numpy.ndarray): A 2D numpy array representing the adjacency matrix.
    Returns: networkx.Graph: A NetworkX graph object.
    """

    assert method in ["adjacency", "laplacian"], "> method should be either 'adjacency' or 'laplacian'"

    match method:
        case "adjacency":
            """
            convert a binary matrix representing a 2D fractal image into a NetworkX graph.
            - nodes are pixels with value 1, and edges connect neighboring pixels.
            - check neighbors (4-connectivity)
            """
            G = nx.Graph()
            rows, cols = matrix.shape
            for i in range(rows):
                for j in range(cols):
                    if matrix[i, j]:
                        G.add_node((i, j))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < rows and 0 <= nj < cols and matrix[ni, nj]:
                                G.add_edge((i, j), (ni, nj))

        case "laplacian":
            if matrix.shape[0] != matrix.shape[1]:
                previous_size = matrix.shape
                matrix = np.pad(matrix, ((0, max(matrix.shape) - matrix.shape[0]), (0, max(matrix.shape) - matrix.shape[1])))
                print(f"> warning: Laplacian matrix should be square. padded matrix: {previous_size} -> {matrix.shape}")
            G = nx.from_numpy_array(matrix)
            G.remove_nodes_from(list(nx.isolates(G)))

    if save_as:
        save_as = save_as if save_as.endswith(".png") else save_as + ".png"
        fig = plt.figure(figsize=(12, 5))

        # Plot matrix in first subplot
        ax1 = fig.add_subplot(121)
        ax1.imshow(matrix, cmap="viridis")
        ax1.set_title("Original Matrix")
        ax1.axis("off")

        # Plot 3D network graph
        ax2 = fig.add_subplot(122, projection="3d")
        pos = nx.spring_layout(G, dim=3)

        # Draw edges
        edge_pos = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        for edge in edge_pos:
            ax2.plot3D(edge[:, 0], edge[:, 1], edge[:, 2], "gray", alpha=0.6)

        # Draw nodes
        node_pos = np.array([pos[node] for node in G.nodes()])
        ax2.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2], c="blue", s=20, alpha=0.6)

        ax2.set_title("3D Network Graph")

        plt.tight_layout()
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.close()

    return G
