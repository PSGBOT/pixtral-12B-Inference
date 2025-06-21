import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools as it
import cv2


def show_graph(G, src_img_path, mask_path):
    # Load images
    src_img = cv2.imread(src_img_path)
    mask_img = cv2.imread(mask_path)

    # Resize src_img to mask size
    if src_img is not None and mask_img is not None:
        mask_h, mask_w, _ = mask_img.shape
        src_img_resized = cv2.resize(src_img, (mask_w, mask_h))
        # Decrease contrast (alpha < 1.0) and increase brightness (beta > 0)
        # A common range for alpha is 0.5-1.5 and for beta is 0-100
        alpha = 0.5  # Contrast control (1.0-3.0)
        beta = 100  # Brightness control (0-100)
        src_img_adjusted = cv2.convertScaleAbs(src_img_resized, alpha=alpha, beta=beta)
    else:
        print("Error: Could not load source image or mask image.")
        return

    # Create a figure and an axes to draw on
    fig, ax = plt.subplots()
    ax.imshow(
        cv2.cvtColor(src_img_adjusted, cv2.COLOR_BGR2RGB)
    )  # Display the adjusted image

    node_sizes = [100 + 20 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = list(range(2, M + 2))  # Convert range to list
    edge_alphas = [(5 + i) / (M + 18) for i in range(M)]
    cmap = plt.cm.plasma
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.1] * 4)]
    nx.draw_networkx_nodes(
        G,
        pos=nx.get_node_attributes(G, "pos"),
        node_size=node_sizes,
        node_color="indigo",
        ax=ax,  # Draw on the specific axes
    )
    edges = nx.draw_networkx_edges(
        G,
        pos=nx.get_node_attributes(G, "pos"),
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=20,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=5,
        connectionstyle=connectionstyle,
        ax=ax,  # Draw on the specific axes
    )
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.show()


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
