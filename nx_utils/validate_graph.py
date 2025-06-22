import networkx as nx
from itertools import combinations

def detect_invalid_kr(G):
    pass


def detect_cyclic_kr(G):
    """ INPUT: NetworkX graph
        OUTPUT: True for has cycle
    """
    try:
        cycle = nx.find_cycle(G)
        return True
    except nx.NetworkXNoCycle:
        return False


def detect_conflict_kr(G):
    pass


def detect_redundancy_kr(G):
    edge_keys = G.edges(keys=True)
    edge_groups = {}

    for u, v, key in edge_keys:
        node_pair = tuple(sorted((u, v)))
        if node_pair not in edge_groups:
            edge_groups[node_pair] = []
        else:
            for edge in edge_groups[node_pair]:
                _, _, key1 = edge
                if key1 == key:
                    return True
        edge_groups[node_pair].append((u, v, key))

    
    return False

