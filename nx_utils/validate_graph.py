import cv2
import numpy as np
import networkx as nx
import os

def detect_cyclic_kr(G, CAT):
    # for all
    # A->B->C->A
    # remove?
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break  # No more cycles

        worst_index = -1
        worst_edge = None

        for u, v in cycle:
            edge_attributes = G.get_edge_data(u, v)

            # get relation type
            joint_type = edge_attributes.get("joint_type", "unknown")
            appendable = ["revolute", "prismatic", "spherical"]
            if joint_type in appendable:
                control_type = edge_attributes.get("controllable")
                joint_type = f"{joint_type}-{control_type}"

            index = CAT.index(joint_type)
            if index > worst_index:
                worst_index = index
                worst_edge = (u, v)

        if worst_edge is not None:
            worst_u, worst_v = worst_edge
            G.remove_edge(worst_u, worst_v)
        
# Deal with cycles with reversible "fixed"
#    while True:
#        try:
#            cycle = nx.find_cycle(G, orientation="ignore")

    return G

def detect_conflict_kr(G, CAT):
    # for all
    # A->B A->B or more, remove until there's at most one edge 

    for part1, part2 in G.nodes():
        all_edges = []
        if G.has_edge(part1, part2):
            for key, attributes in G[part1][part2].items():
                all_edges.append((part1, part2, key, attributes))
        if G.has_edge(part2, part1):
            for key, attributes in G[part2][part1].items():
                all_edges.append((part2, part1, key, attributes))
        if len(all_edges) <= 1: # no conflict
            continue

        edges_to_remove = []
        best_index = float('inf')
        best_edge = None

        for an_edge in all_edges:
            _, _, _, edge_attributes = an_edge
            # get relation type
            joint_type = edge_attributes.get("joint_type", "unknown")
            appendable = ["revolute", "prismatic", "spherical"]
            if joint_type in appendable:
                control_type = edge_attributes.get("controllable")
                joint_type = f"{joint_type}-{control_type}"

            index = CAT.index(joint_type)
            if index < best_index:
                    best_index = index
                    best_edge = an_edge
        
        # Mark all others for removal
        for edge in all_edges:
            if edge != best_edge:
                edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)
    return G

def get_margin(mask_u, mask_v): # GPT
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
    for part1, part2 in G.nodes(): # margin between masks, remove max margin relation
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
