import networkx as nx


def read_rel_as_nx(kr_list, pos_dict, all_rel=False):
    # read the kinematic relationship as networkx graph
    G = nx.MultiDiGraph()
    for rel in kr_list:
        part0_id = rel[0]
        part1_id = rel[1]
        rel_data = rel[2]
        kinematic_joints = rel_data.get("kinematic_joints", [])
        part0_func = rel_data.get("part0_function", [])
        part1_func = rel_data.get("part1_function", [])
        part0_desc = rel_data.get("part0_desc", "")
        part1_desc = rel_data.get("part1_desc", "")
        for kj in kinematic_joints:
            root_part = kj.get("root")

            if not part0_id or not part1_id:
                continue

            edge_attributes = {}
            # assign joint type
            joint_type = kj.get("joint_type", "unknown")
            if joint_type in ["unrelated", "unknown"] and not all_rel:
                continue  # Discard "unrelated" and "unknown" relations
            edge_attributes["joint_type"] = joint_type
            if "controllable" in kj:
                edge_attributes["controllable"] = kj["controllable"]
            edge_attributes["root"] = root_part

            # assign node function per edge
            edge_attributes["part0_function"] = part0_func
            edge_attributes["part1_function"] = part1_func

            # assign node description per edge
            edge_attributes["part0_desc"] = part0_desc
            edge_attributes["part1_desc"] = part1_desc

            if not G.has_node(part0_id):
                pos = pos_dict.get(part0_id, [0, 0])
                G.add_node(part0_id, pos=(pos[0], pos[1]))
            if not G.has_node(part1_id):
                pos = pos_dict.get(part1_id, [0, 0])
                G.add_node(part1_id, pos=(pos[0], pos[1]))

            if root_part == "0":
                G.add_edge(part1_id, part0_id, **edge_attributes)
            elif root_part == "1":
                G.add_edge(part0_id, part1_id, **edge_attributes)
            # print(part1_id, part2_id, edge_attributes)
    return G


# find root? after removing cyclics
# weight? -> delete cycle -> find root -> prune -> output config.json
