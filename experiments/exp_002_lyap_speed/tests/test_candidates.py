"""Regression coverage for numerical fidelity of each named speed candidate."""

import numpy as np
import pytest

from experiments.exp_002_lyap_speed.src.modules.candidates import neighbours_kdtree, neighbours_parallel, neighbours_scalar
from experiments.exp_002_lyap_speed.src.modules.kernels import neighbours_original
from experiments.exp_002_lyap_speed.src.modules.verification import cases, verify_case


@pytest.mark.parametrize("name,x,parameters", cases(), ids=[case[0] for case in cases()])
def test_public_results_and_neighbour_indices_match(name, x, parameters):
    """Test the full divergence trajectory, exact neighbour index and final slope."""
    rows = verify_case(name, x, parameters)
    assert len(rows) == 3
    assert all(row["divergence_error"] <= 1e-12 and row["exponent_error"] <= 1e-12 for row in rows)


@pytest.mark.parametrize("search", [neighbours_scalar, neighbours_parallel, neighbours_kdtree])
def test_tied_initial_distance_preserves_future_trajectory(search):
    """Equal initial points with different futures must keep the first eligible index."""
    y = np.array([[0.0], [5.0], [1.0], [8.0], [-1.0], [99.0], [3.0], [-4.0]])
    expected = neighbours_original(y, 1, 2)
    assert expected[0] == 2
    np.testing.assert_array_equal(search(y, 1, 2), expected)


def test_kdtree_excludes_neighbours_without_full_future():
    """Do not choose a close point from the final trajectory_len-1 rows."""
    y = np.array([[0.0], [5.0], [8.0], [11.0], [13.0], [0.0]])
    expected = neighbours_original(y, 0, 3)
    assert np.max(expected) < 4
    np.testing.assert_array_equal(neighbours_kdtree(y, 0, 3), expected)
