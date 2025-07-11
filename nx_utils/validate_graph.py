import cv2
import numpy as np
import os


def detect_cyclic_kr(G, dir):
    # for all
    # A->B->C->A
    # remove?

    invalid_edges = []

    G.remove_edges_from(invalid_edges)
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


def detect_redundancy_kr(G, dir):
    # for all
    # A->B->C A->C
    # remove A->C
    invalid_edges = []
    for part1, part2 in G.edges(): # margin between masks, remove max margin relation
        for part3 in G.nodes():
            if part3 != part1 and part3 != part2:
                if G.has_edge(part1, part3) and G.has_edge(part3, part2):
                    invalid_edges.append((part1, part2))
                    break
    G.remove_edges_from(invalid_edges)
    return G
