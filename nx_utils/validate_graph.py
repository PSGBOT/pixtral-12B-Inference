import re
import cv2
import numpy as np
import networkx as nx
import os
from collections import defaultdict, deque

appendable_joint_types = ["revolute", "prismatic", "spherical"]


def detect_cyclic_kr(G, CAT):
    # for all
    # A->B->C->A
    turn = 0
    while turn < 20:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            print("cycle found")
            turn += 1
        except nx.NetworkXNoCycle:
            break  # No more cycles

        worst_index = -1
        worst_pair = None

        for u, v, key, _ in cycle:
            edge_data = G.get_edge_data(u, v, key)
            best_index = float("inf")

            # get relation type
            joint_type = edge_data.get("joint_type")
            if joint_type in appendable_joint_types:
                control_type = edge_data.get("controllable")
                joint_type = f"{joint_type}-{control_type}"
            index = CAT.index(joint_type)
            if index > worst_index:
                worst_index = index
                worst_pair = (u, v, key)

        if worst_pair:
            G.remove_edge(worst_pair[0], worst_pair[1], worst_pair[2])

    assert turn < 20
    return G


def detect_conflict_kr(G, CAT_index):
    for part1 in G.nodes():
        for part2 in G.nodes():
            # same direction
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
            for e_i in all_edges:
                _, _, _, ea_i = e_i
                if ea_i.get("joint_type") == "unknown":
                    keep_edge = e_i
                    for e_j in all_edges:
                        if e_j != keep_edge:
                            G.remove_edge(e_j[0], e_j[1], e_j[2])

            # iff "fixed", keep 1 "fixed"
            if all(attr.get("joint_type") == "fixed" for _, _, _, attr in all_edges):
                # Pick the first "fixed" edge to keep
                keep_edge = all_edges[0]
                for e_j in all_edges:
                    if e_j != keep_edge:
                        G.remove_edge(e_j[0], e_j[1], e_j[2])
                continue

            valid_edges = []  # skip "supported", "flexible", "unrelated"
            skippable_joint_types = ["supported", "flexible", "unrelated"]
            for e_i in all_edges:
                _, _, _, ea_i = e_i
                joint_type = ea_i.get("joint_type")
                if joint_type not in skippable_joint_types:
                    valid_edges.append(e_i)
            valid_remove = []

            if len(all_edges) <= 1:
                continue
            else:
                # delete "fixed" if other edge is "static"
                for e_i in valid_edges:
                    _, _, _, ea_i = e_i
                    if (
                        ea_i.get("controllable") == "static"
                        and ea_i.get("joint_type") != "fixed"
                    ):  # other kind of edge with static attribute
                        for e_j in valid_edges:
                            _, _, _, ea_j = e_j
                            if ea_j.get("joint_type") == "fixed":
                                G.remove_edge(e_j[0], e_j[1], e_j[2])
                                valid_remove.append(e_j)
                for remove in valid_remove:
                    valid_edges.remove(remove)
                # merge
                fixed_edge = next(
                    (
                        edge
                        for edge in valid_edges
                        if edge[3].get("joint_type") == "fixed"
                    ),
                    None,
                )
                other_edges = [
                    edge
                    for edge in valid_edges
                    if edge[3].get("joint_type") != "fixed"
                    and edge[3].get("controllable") != "static"
                ]
                for other_edge in other_edges:
                    if fixed_edge and other_edge:
                        _, _, key, other_attr = other_edge
                        merged_attr = dict(other_attr)
                        merged_attr["controllable"] = "static"
                        G[part1][part2][key].update(merged_attr)
                if fixed_edge:
                    G.remove_edge(fixed_edge[0], fixed_edge[1], fixed_edge[2])

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

            best_index_forward = float("inf")
            best_index_backward = float("inf")

            for e_i in forward_edges:
                _, _, _, ea_i = e_i
                # get relation type
                joint_type = ea_i.get("joint_type")
                if joint_type in appendable_joint_types:
                    control_type = ea_i.get("controllable")
                    joint_type = f"{joint_type}-{control_type}"

                index = CAT_index.index(joint_type)
                if index < best_index_forward:
                    best_index_forward = index

            for e_i in backward_edges:
                _, _, _, ea_i = e_i
                # get relation type
                joint_type = ea_i.get("joint_type")
                if joint_type in appendable_joint_types:
                    control_type = ea_i.get("controllable")
                    joint_type = f"{joint_type}-{control_type}"

                index = CAT_index.index(joint_type)
                if index < best_index_backward:
                    best_index_backward = index

            if best_index_forward < best_index_backward:
                G.remove_edges_from(backward_edges)
            else:
                G.remove_edges_from(forward_edges)

    return G


def get_margin(mask_u, mask_v):
    """
    Calculates the minimum Euclidean distance between two binary masks.

    Args:
        mask_u (numpy.ndarray): First binary mask (2D array, non-zero for foreground).
        mask_v (numpy.ndarray): Second binary mask (2D array, non-zero for foreground).

    Returns:
        float: The minimum Euclidean distance between the two masks.
               Returns float('inf') if either mask is empty.
    """
    # Ensure masks are boolean or uint8 for distanceTransform
    mask_u_bool = mask_u > 0
    mask_v_bool = mask_v > 0

    if not np.any(mask_u_bool) or not np.any(mask_v_bool):
        return float("inf")  # No valid mask area, treat as max distance

    # Check if masks overlap
    if np.any(mask_u_bool & mask_v_bool):
        return 0.0  # Masks overlap, distance is 0

    # For disjoint masks, compute distance transform of the inverse of mask_u
    # This gives us the distance from each pixel TO the nearest mask_u pixel
    mask_u_inverse = (~mask_u_bool).astype(np.uint8)
    dist_transform_u = cv2.distanceTransform(mask_u_inverse, cv2.DIST_L2, 5)

    # Get the distances from pixels in mask_v to the nearest pixel in mask_u
    # We only care about the distances at the locations of white pixels in mask_v
    distances_at_v_pixels = dist_transform_u[mask_v_bool]

    # The minimum of these distances is the margin
    return np.min(distances_at_v_pixels)


def find_kinematic_root(G):
    # 方法 1：找出度为 0 的节点
    root_candidates = [n for n in G.nodes if G.out_degree(n) == 0]
    if len(root_candidates) == 1:
        return root_candidates[0], root_candidates
    elif len(root_candidates) > 1:
        print(f"⚠️ 警告：存在多个出度为 0 的节点，使用传播法找 root: {root_candidates}")

    # 方法 2：反向传播：从叶子节点向上累积值
    node_value = defaultdict(int)
    leaves = [n for n in G.nodes if G.in_degree(n) == 0]
    queue = deque(leaves)

    for leaf in leaves:
        node_value[leaf] = 1

    while queue:
        child = queue.popleft()
        for parent in G.successors(child):
            node_value[parent] += node_value[child]
            queue.append(parent)

    # 找累计值最大的节点
    max_value = max(node_value.values())
    best_roots = [n for n, v in node_value.items() if v == max_value]

    if len(best_roots) == 1:
        return best_roots[0], root_candidates
    else:
        print(f"⚠️ 警告：传播后仍有多个 root 候选，返回其中之一: {best_roots}")
        return best_roots[0], root_candidates


def graph_swap_dir(G: nx.MultiDiGraph, u: str, v: str, key):
    attr = G.get_edge_data(u, v, key)
    new_attr = attr.copy()
    root_part = attr.get("root")
    part0_func = attr.get("part0_function")
    part1_func = attr.get("part1_function")
    part0_desc = attr.get("part0_desc")
    part1_desc = attr.get("part1_desc")
    if root_part == "0":
        new_attr["root"] = "1"
    elif root_part == "1":
        new_attr["root"] = "0"
    # assign node function per edge
    new_attr["part0_function"] = part1_func
    new_attr["part1_function"] = part0_func

    # assign node description per edge
    new_attr["part0_desc"] = part1_desc
    new_attr["part1_desc"] = part0_desc

    G.remove_edge(u, v, key)
    G.add_edge(v, u, **new_attr)


def graph_to_tree(G: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, str]:
    root, root_candidates = find_kinematic_root(G)
    for candidate in root_candidates:
        if candidate is root:
            continue
        else:
            # get all the directed edge point to the candidate root
            incoming_edges = list(G.in_edges(candidate, keys=True))
            for other, candy, key, attri in incoming_edges:
                try:
                    # reorient the edge
                    graph_swap_dir(G, other, candy, key)
                    cycle = nx.find_cycle(G, orientation="original")
                    print("cycle found, restore re-oriention")
                    graph_swap_dir(G, candy, other, key)
                except Exception as e:
                    print("no cycle found, pass the re-oriention")
                    break
                raise Exception(
                    "all the edges point to the candidate cannot be re-oriented"
                )
    root, root_candidates = find_kinematic_root(G)
    if len(root_candidates) > 1:
        raise Exception("invalid re-oriented graph")

    return G, root


def leave_to_root_list(G: nx.MultiDiGraph):
    leaves = [n for n in G.nodes if G.in_degree(n) == 0]
    queue = deque(leaves)

    while queue:
        child = queue.popleft()
        for parent in G.successors(child):
            queue.append(parent)
            if parent in leaves:
                leaves.remove(parent)
            leaves.append(parent)
    return leaves


def detect_redundancy_kr(G: nx.MultiDiGraph, root: str, dir):
    """
    Detect and remove redundant edges in a kinematic tree structure.
    For each node (except root), find all paths to the root and remove redundant edges
    based on the maximum margin between masks.

    Args:
        G: NetworkX MultiDiGraph representing the kinematic structure
        root: The root node of the tree
        dir: Directory containing mask images for margin calculation

    Returns:
        G: Graph with redundant edges removed
    """
    edges_to_remove = []

    node_list = leave_to_root_list(G)
    for node in node_list:
        if node == root:
            continue

        # Check if node has multiple parents (potential redundancy)
        parents = list(G.successors(node))
        if len(parents) <= 1:
            continue

        mask_node_path = os.path.join(dir, f"{node}.png")
        mask_node = cv2.imread(mask_node_path, cv2.IMREAD_GRAYSCALE)

        margin_dict = {}
        for parent in parents:
            mask_parent_path = os.path.join(dir, f"{parent}.png")
            mask_parent = cv2.imread(mask_parent_path, cv2.IMREAD_GRAYSCALE)

            # Calculate margin between node and parent masks
            margin = get_margin(mask_node, mask_parent)
            margin_dict[parent] = margin
        print(margin_dict)

        # If we have multiple parents with calculated margins, keep only the one with minimum margin
        if len(margin_dict) > 1:
            # Find parent with minimum margin (closest relationship)
            min_margin_parent = min(margin_dict.items(), key=lambda x: x[1])[0]

            # Remove all edges from node to all parents except the one with minimum margin
            for parent in margin_dict:
                if parent != min_margin_parent and G.has_edge(node, parent):
                    edges_between = list(G[node][parent].keys())
                    for key in edges_between:
                        edges_to_remove.append((node, parent, key))

    # Remove redundant edges
    for u, v, key in edges_to_remove:
        if G.has_edge(u, v, key):
            G.remove_edge(u, v, key)
            print(f"Removed redundant edge: {u} -> {v} (key: {key})")

    return G


def detect_redundancy_kr_sample(G: nx.MultiDiGraph, root: str, dir):
    """
    Detect and remove redundant edges in a kinematic tree structure.
    For each node (except root), find all paths to the root and remove redundant edges
    based on the maximum margin between masks.

    Args:
        G: NetworkX MultiDiGraph representing the kinematic structure
        root: The root node of the tree
        dir: Directory containing mask images for margin calculation

    Returns:
        G: Graph with redundant edges removed
    """
    edges_to_remove = []

    # For each node (except root), find paths to root
    for node in G.nodes():
        if node == root:
            continue

        try:
            # Find all simple paths from node to root
            all_paths = list(nx.all_simple_paths(G, node, root))

            if len(all_paths) <= 1:
                continue  # No redundancy if only one path exists

            # Collect all edges involved in these paths
            path_edges = set()
            for path in all_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # Get all edges between u and v (in case of MultiDiGraph)
                    if G.has_edge(u, v):
                        for key in G[u][v]:
                            path_edges.add((u, v, key))

            # Find redundant edges by checking for triangular relationships
            # A->B->C and A->C, where A->C is potentially redundant
            for path in all_paths:
                if len(path) < 2:
                    continue

                # Check each segment of the path for potential shortcuts
                for i in range(len(path) - 2):
                    start_node = path[i]
                    intermediate_node = path[i + 1]
                    end_node = path[i + 2]

                    # Check if there's a direct edge from start to end (shortcut)
                    if G.has_edge(start_node, end_node):
                        # We have a potential redundancy: start->intermediate->end vs start->end
                        edges_in_triangle = [
                            (start_node, intermediate_node),
                            (intermediate_node, end_node),
                            (start_node, end_node),
                        ]

                        # Calculate margins for each edge to determine which to remove
                        max_margin = -1
                        edge_to_remove = None

                        for edge_u, edge_v in edges_in_triangle:
                            if not G.has_edge(edge_u, edge_v):
                                continue

                            try:
                                mask_u_path = os.path.join(dir, f"{edge_u}.png")
                                mask_v_path = os.path.join(dir, f"{edge_v}.png")

                                if os.path.exists(mask_u_path) and os.path.exists(
                                    mask_v_path
                                ):
                                    mask_u = cv2.imread(
                                        mask_u_path, cv2.IMREAD_GRAYSCALE
                                    )
                                    mask_v = cv2.imread(
                                        mask_v_path, cv2.IMREAD_GRAYSCALE
                                    )

                                    if mask_u is not None and mask_v is not None:
                                        margin = get_margin(mask_u, mask_v)
                                        if margin > max_margin:
                                            max_margin = margin
                                            # Store all keys for this edge pair
                                            if G.has_edge(edge_u, edge_v):
                                                for key in G[edge_u][edge_v]:
                                                    edge_to_remove = (
                                                        edge_u,
                                                        edge_v,
                                                        key,
                                                    )
                            except Exception as e:
                                print(
                                    f"Error processing masks for {edge_u}->{edge_v}: {e}"
                                )
                                continue

                        # Add the edge with maximum margin to removal list
                        if edge_to_remove and edge_to_remove not in edges_to_remove:
                            edges_to_remove.append(edge_to_remove)

        except nx.NetworkXNoPath:
            # No path exists from this node to root, skip
            continue
        except Exception as e:
            print(f"Error processing node {node}: {e}")
            continue

    # Remove redundant edges
    for u, v, key in edges_to_remove:
        if G.has_edge(u, v, key):
            G.remove_edge(u, v, key)
            print(f"Removed redundant edge: {u} -> {v} (key: {key})")

    return G


def detect_redundancy_kr_deprecated(G, dir):
    # for all
    # A->B->C A->C
    # remove A->C
    redundant = []
    edges_to_remove = []
    for part1, part2 in G.nodes():  # margin between masks, remove max margin relation
        for part3 in G.nodes():
            if part3 != part1 and part3 != part2:
                if (
                    G.has_edge(part1, part3)
                    and G.has_edge(part3, part2)
                    and G.has_edge(part1, part3)
                ):
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
