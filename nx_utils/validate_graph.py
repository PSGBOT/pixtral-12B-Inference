import cv2
import numpy as np
import networkx as nx
import os
from collections import defaultdict # for dict

appendable_joint_types = ["revolute", "prismatic", "spherical"]

def detect_cyclic_kr(G, CAT):
    # for all
    # A->B->C->A
    # remove?
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break  # No more cycles

        worst_best_index = -1
        worst_pair = None

        for u, v in cycle:
            edges = G.get_edge_data(u, v)
            best_index = float("inf")

            # get relation type
            for key, edge_attributes in edges.items():
                joint_type = edge_attributes.get("joint_type")
                if joint_type in appendable_joint_types:
                    control_type = edge_attributes.get("controllable")
                    joint_type = f"{joint_type}-{control_type}"
                index = CAT.index(joint_type)
                if index < best_index:
                    best_index = index

            if best_index > worst_best_index:
                worst_best_index = best_index
                worst_pair = (u, v)
            
        # remove edges between the worst pair of nodes
        all_keys = list(G[worst_pair[0]][worst_pair[1]].keys())
        for key in all_keys:
            G.remove_edge(G[worst_pair[0]], G[worst_pair[1]], key)

    return G

def detect_conflict_kr(G, CAT):

    # same direction
    for part1 in G.nodes():
        for part2 in G.nodes():
            if part1 == part2:
                continue
            if not G.has_edge(part1, part2):
                continue
            
            all_edges = []
            for key, attributes in G[part1][part2].items():
                all_edges.append((part1, part2, key, attributes))

            if len(all_edges) <= 1:
                continue

            # if "unknown" detected, discard all others
            for an_edge in all_edges:
                _, _, _, edge_attributes = an_edge
                if edge_attributes.get("joint_type") == "unknown":
                    keep_edge = an_edge
                    for edge in all_edges:
                        if edge != keep_edge:
                            G.remove_edge(edge)

            # iff "fixed", keep 1 "fixed"
            if all(attr.get("joint_type") == "fixed" for _, attr in all_edges):
                # Pick the first "fixed" edge to keep
                keep_edge = all_edges[0]
                for edge in all_edges:
                    if edge != keep_edge:
                        G.remove_edge(edge)
                continue

            valid_edges = [] # skip "supported", "flexible", "unrelated"
            skippable_joint_types = ["supported", "flexible", "unrelated"]
            for an_edge in all_edges:
                _, _, _, edge_attributes = an_edge
                joint_type = edge_attributes.get("joint_type")
                if joint_type not in skippable_joint_types:
                    valid_edges.append(an_edge)

            if len(valid_edges) <= 1:
                continue
            # else try to merge
            for an_edge in valid_edges:
                _, _, _, edge_attributes = an_edge
                if edge_attributes.get("joint_type") == "static":
                    for edge in valid_edges:
                        _, _, _, edge_attributes = an_edge
                        if edge_attributes.get("joint_type") == "fixed":
                            G.remove_edge(edge)
            # merge This part is GPT, rework needed
            fixed_edge = next((e for e in valid_edges if e[3].get("joint_type") == "fixed"), None)
            other_edge = next(
                (e for e in valid_edges if e[3].get("joint_type") != "fixed" and e[3].get("controllable") != "static"),
                None
            )
            if fixed_edge and other_edge:
                _, _, _, other_attr = other_edge
                merged_attr = dict(other_attr)
                merged_attr["controllable"] = "fixed"

                # Remove all current valid edges
                for edge in valid_edges:
                    G.remove_edge(edge[0], edge[1], edge[2])

                # Add the merged edge
                G.add_edge(part1, part2, key="merged", **merged_attr)
                continue

        # opposite direction
        forward_edges = []
        backward_edges = []
        if G.has_edge(part1, part2):
            for key, attributes in G[part1][part2].items():
                forward_edges.append((part1, part2, key, attributes))
        else:
            continue
        if G.has_edge(part2, part1):
            for key, attributes in G[part2][part1].items():
                backward_edges.append((part2, part1, key, attributes))
        else:
            continue

        best_index_forward = float('inf')
        best_index_backward = float('inf')

        for an_edge in forward_edges:
            _, _, _, edge_attributes = an_edge
            # get relation type
            joint_type = edge_attributes.get("joint_type")
            if joint_type in appendable_joint_types:
                control_type = edge_attributes.get("controllable")
                joint_type = f"{joint_type}-{control_type}"

            index = CAT.index(joint_type)
            if index < best_index_forward:
                    best_index_forward = index

        for an_edge in backward_edges:
            _, _, _, edge_attributes = an_edge
            # get relation type
            joint_type = edge_attributes.get("joint_type")
            if joint_type in appendable_joint_types:
                control_type = edge_attributes.get("controllable")
                joint_type = f"{joint_type}-{control_type}"

            index = CAT.index(joint_type)
            if index < best_index_backward:
                    best_index_backward = index
        
        if best_index_forward < best_index_backward:
            G.remove_edges_from(backward_edges)
        else:
            G.remove_edges_from(forward_edges)

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

def graph_to_tree(G, CAT):
    best_edges = []
    for u, v, key, edge_attributes in G.edges():
        joint_type = edge_attributes.get("joint_type", "unknown")
        if joint_type in appendable_joint_types:
            control_type = edge_attributes.get("controllable")
            joint_type = f"{joint_type}-{control_type}"
        weight = CAT.index(joint_type)
    
# needs re-org from here down
        a, b = sorted((u, v))
        if (a, b) not in best_edges or best_edges[(a, b)][0] > weight:
            best_edges[(a, b)] = (weight, joint_type)

    UG = nx.Graph()
    for (u, v), (w, _) in best_edges.items():
        UG.add_edge(u, v, weight=w)
    root = min(UG.nodes, key=lambda n: sum(nx.single_source_dijkstra_path_length(UG, n).values()))
    visited = set()
    queue = [root]
    parent_map = {}
    while queue:
        current = queue.pop(0)
        visited.add(current)
        for neighbor in UG.neighbors(current):
            if neighbor in visited:
                continue
            parent_map[neighbor] = current
            queue.append(neighbor)
    edges_to_add = []
    edges_to_remove = []

    for child, parent in parent_map.items():
        # Check if there are edges from child to parent (wrong direction)
        if G.has_edge(child, parent):
            for k, data in list(G[child][parent].items()):
                joint_type = data.get("joint_type", "")
                if joint_type == "fixed":
                    # Reorient fixed edge to go parent -> child
                    edges_to_remove.append((child, parent, k))
                    edges_to_add.append((parent, child, data))  # reversed
        # Check if parent already points to child (correct direction)

    # Apply reorientation
    for u, v, k in edges_to_remove:
        G.remove_edge(u, v, k)
    for u, v, data in edges_to_add:
        G.add_edge(u, v, **data)
        
    return G, root

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

# in conflict: merge-able edges in the same direction
# e.g. same direction: "fixed"+"...-revolute"->"...-static" / "fixed"+"revolute-static"->"revolute-static"

# in cyclic: *multiple edges between a pair of nodes is possible (weight follows the highest-weight edge)
# in cyclic: "fixed" direction problem?

# in redundancy: find the root of the tree and substitute the double for loop with pathing / change "fixed" directions to make all edges direct to (from?) root
# in redundancy: margin between mask detection

# TEST
# optimize: get_relation_index could be an independent function