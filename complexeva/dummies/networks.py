#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

"""
all the networkx class dummy implementations
"""

import numpy as np
import networkx as nx


class WebFractalNetwork:
    """
    Generates a pseudo-fractal network using an iterative method.

    Usage:
    web_fractal = WebFractalNetwork(complexity=3)
    network = web_fractal.network
    """

    def __init__(self, complexity=3):
        self.complexity = complexity
        self.network = self._generate_network()
        self.diameter = nx.diameter(self.network)

    def _generate_network(self):
        """
        Generates the pseudo-fractal network.
        """
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        for _ in range(self.complexity):
            G = self._fractal_step(G)
        return G

    def _fractal_step(self, G):
        """
        Performs one iteration of the fractal construction.
        """
        new_edges = []
        max_node = max(G.nodes())
        for u, v in G.edges():
            new_node = max_node + 1
            max_node += 1
            new_edges.extend([(new_node, u), (new_node, v)])
        G.add_edges_from(new_edges)
        return G


class SierpinskiFractal:
    """
    Generates a Sierpinski gasket fractal network.

    Usage:
    sierpinski = SierpinskiFractal(recursion_depth=3)
    network = sierpinski.network
    """

    def __init__(self, recursion_depth=3):
        self.recursion_depth = recursion_depth
        self.points = {}
        self.edges = set()
        self.point_index = 0
        self.network = self._generate_network()

    def _generate_network(self):
        """
        Generates the Sierpinski gasket fractal network.
        """
        vertices = [(0, 0), (0.5, np.sqrt(3) / 2), (1, 0)]
        self._sierpinski_recursive(vertices, self.recursion_depth)
        return self._build_graph()

    def _sierpinski_recursive(self, vertices, depth):
        """
        Recursively generates the Sierpinski gasket fractal.
        """

        def add_triangle(v1, v2, v3, current_depth):
            if current_depth == 0:
                for v in [v1, v2, v3]:
                    if v not in self.points:
                        self.points[v] = self.point_index
                        self.point_index += 1
                i1, i2, i3 = self.points[v1], self.points[v2], self.points[v3]
                self.edges.update([(i1, i2), (i2, i3), (i3, i1)])
            else:
                m1 = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
                m2 = ((v2[0] + v3[0]) / 2, (v2[1] + v3[1]) / 2)
                m3 = ((v3[0] + v1[0]) / 2, (v3[1] + v1[1]) / 2)
                add_triangle(v1, m1, m3, current_depth - 1)
                add_triangle(m1, v2, m2, current_depth - 1)
                add_triangle(m3, m2, v3, current_depth - 1)

        add_triangle(vertices[0], vertices[1], vertices[2], depth)

    def _build_graph(self):
        """
        Builds the NetworkX graph from the generated points and edges.
        """
        G = nx.Graph()
        for coord, idx in self.points.items():
            G.add_node(idx, pos=coord)
        G.add_edges_from(self.edges)
        return G


if __name__ == "__main__":
    network = SierpinskiFractal(recursion_depth=7)
    # network = WebFractalNetwork(complexity=4)
    # handler = FractalHandlerNetworkX(network.network)
    # handler.compute_fractal_dimensions(box_sizes="auto", iterations=10)
