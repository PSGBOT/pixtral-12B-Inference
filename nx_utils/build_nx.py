import networkx as nx


def read_rel_as_nx(kr_list, pos_dict):
    # read the kinematic relationship as networkx graph
    G = nx.MultiDiGraph()
    for rel in kr_list:
        part1_id = rel[0]
        part2_id = rel[1]
        rel_data = rel[2]
        kinematic_joints = rel_data.get("kinematic_joints", [])
        for kj in kinematic_joints:
            root_part = kj.get("root")

            if not part1_id or not part2_id:
                continue

            edge_attributes = {}
            # Assuming one primary joint for simplicity, or combine attributes
            joint_type = kj.get("joint_type", "unknown")
            if joint_type in ["unrelated", "unknown"]:
                continue  # Discard "unrelated" and "unknown" relations
            edge_attributes["joint_type"] = joint_type
            if "controllable" in kj:
                edge_attributes["controllable"] = kj["controllable"]

            if not G.has_node(part1_id):
                pos = pos_dict.get(part1_id, [0, 0])
                G.add_node(part1_id, pos=(pos[0], pos[1]))
            if not G.has_node(part2_id):
                pos = pos_dict.get(part2_id, [0, 0])
                G.add_node(part2_id, pos=(pos[0], pos[1]))

            if root_part == "0":
                G.add_edge(part2_id, part1_id, **edge_attributes)
            elif root_part == "1":
                G.add_edge(part1_id, part2_id, **edge_attributes)
            # print(part1_id, part2_id, edge_attributes)
    return G

# find root? after removing cyclics
# weight? -> delete cycle -> find root -> prune -> output config.json