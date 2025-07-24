from _pytest.config import main
import networkx as nx
import pytest
from nx_utils.validate_graph import graph_swap_dir


def test_graph_swap_dir():
    # Create a MultiDiGraph
    G = nx.MultiDiGraph()

    # Add an edge with attributes
    u = "partA"
    v = "partB"
    key = 0
    attributes = {
        "joint_type": "revolute",
        "controllable": "static",
        "root": "0",
        "part0_function": "rotate",
        "part1_function": "fixed",
        "part0_desc": "base part",
        "part1_desc": "rotating part",
    }
    G.add_edge(u, v, key=key, **attributes)

    print(f"Initial graph: Edges: {G.edges(data=True)}")
    # Ensure the initial edge exists
    assert G.has_edge(u, v, key)
    assert not G.has_edge(v, u)

    # Call the function to swap direction
    graph_swap_dir(G, u, v, key)
    print(f"Graph after swap: {G}")

    # Assert the original edge is removed
    assert not G.has_edge(u, v, key)

    # Assert the new edge exists in the swapped direction
    assert G.has_edge(v, u, key)

    # Get the attributes of the new edge
    swapped_attributes = G.get_edge_data(v, u, key)

    # Assert attributes are correctly swapped
    assert swapped_attributes["joint_type"] == attributes["joint_type"]
    assert swapped_attributes["controllable"] == attributes["controllable"]
    assert swapped_attributes["root"] == "1"  # Root should be swapped from "0" to "1"
    assert swapped_attributes["part0_function"] == attributes["part1_function"]
    assert swapped_attributes["part1_function"] == attributes["part0_function"]
    assert swapped_attributes["part0_desc"] == attributes["part1_desc"]
    assert swapped_attributes["part1_desc"] == attributes["part0_desc"]
    print(f"Swapped attributes for (partB, partA, 0): {swapped_attributes}")

    # Test with root "1"
    G_root1 = nx.MultiDiGraph()
    attributes_root1 = {
        "joint_type": "prismatic",
        "controllable": "movable",
        "root": "1",
        "part0_function": "slide",
        "part1_function": "fixed",
        "part0_desc": "sliding part",
        "part1_desc": "track",
    }
    G_root1.add_edge(u, v, key=key, **attributes_root1)
    graph_swap_dir(G_root1, u, v, key)
    swapped_attributes_root1 = G_root1.get_edge_data(v, u, key)
    print(
        f"Swapped attributes for G_root1 (partB, partA, 0): {swapped_attributes_root1}"
    )
    assert (
        swapped_attributes_root1["root"] == "0"
    )  # Root should be swapped from "1" to "0"

    # Test with no root attribute
    G_no_root = nx.MultiDiGraph()
    attributes_no_root = {
        "joint_type": "spherical",
        "controllable": "static",
        "part0_function": "pivot",
        "part1_function": "fixed",
        "part0_desc": "ball",
        "part1_desc": "socket",
    }
    G_no_root.add_edge(u, v, key=key, **attributes_no_root)
    graph_swap_dir(G_no_root, u, v, key)
    swapped_attributes_no_root = G_no_root.get_edge_data(v, u, key)
    print(
        f"Swapped attributes for G_no_root (partB, partA, 0): {swapped_attributes_no_root}"
    )
    assert (
        "root" not in swapped_attributes_no_root
    )  # Root attribute should remain absent


if __name__ == "__main__":
    test_graph_swap_dir()
