import cv2
import numpy as np
import networkx as nx
import os

def detect_cyclic_kr(G, dir):
    # for all
    # A->B->C->A
    # remove?
    while True:
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break  # No more cycles

        max_margin = -1
        edge_to_remove = None

        for u, v in cycle:
            mask_u_path = os.path.join(dir, f"{u}.png")
            mask_v_path = os.path.join(dir, f"{v}.png")
            mask_u = cv2.imread(mask_u_path, cv2.IMREAD_GRAYSCALE)
            mask_v = cv2.imread(mask_v_path, cv2.IMREAD_GRAYSCALE)

            margin = get_margin(mask_u, mask_v)
            if margin > max_margin:
                max_margin = margin
                edge_to_remove = (u, v)

        # Remove edge with largest margin
        if edge_to_remove and G.has_edge(*edge_to_remove):
            u, v = edge_to_remove
            keys = list(G[u][v].keys())
            for k in keys:
                G.remove_edge(u, v, k) if k is not None else G.remove_edge(u, v)


    return G


def detect_conflict_kr(G, dir):
    # for all
    # A->B A->B
    # if "fixed"&""
    # ball? keep fixed
    # ? remove "fixed" and change "controllable" to "static"

    invalid_edges = []

    G.remove_edges_from(invalid_edges)
    return G

def get_margin(mask_u, mask_v):
    # Get white pixel coordinates in each mask
    u_points = np.column_stack(np.where(mask_u > 0))
    v_points = np.column_stack(np.where(mask_v > 0))

    if len(u_points) == 0 or len(v_points) == 0:
        return float('inf')  # No valid mask area, treat as max distance

    # Compute all pairwise distances and return the minimum one
    dists = np.linalg.norm(u_points[:, np.newaxis] - v_points[np.newaxis, :], axis=2)
    return np.min(dists)

def detect_redundancy_kr(G, dir):
    # for all
    # A->B->C A->C
    # remove A->C
    redundant = []
    edges_to_remove = []
    for part1, part2 in G.edges(): # margin between masks, remove max margin relation
        for part3 in G.nodes():
            if part3 != part1 and part3 != part2:
                if G.has_edge(part1, part3) and G.has_edge(part3, part2) and G.has_edge(part1, part3):
                    redundant = [(part1, part3), (part3, part2), (part1, part2)]

                    max_margin = -1
                    longest_edge = None
                    for edge_u, edge_v in redundant:
                        mask_u_path = os.path.join(dir, f"{edge_u}.png")
                        mask_v_path = os.path.join(dir, f"{edge_v}.png")
                        mask_u = cv2.imread(mask_u_path, cv2.IMREAD_GRAYSCALE)
                        mask_v = cv2.imread(mask_v_path, cv2.IMREAD_GRAYSCALE)

                        margin = get_margin(mask_u, mask_v)
                        if margin > max_margin:
                            max_margin = margin
                            longest_edge = (edge_u, edge_v)

                    edges_to_remove.append(longest_edge)

    G.remove_edges_from(edges_to_remove)
    return G
