import cv2
import numpy as np
import os

def detect_invalid_kr(G, dir):
    # if A and B are not connected, the relation detected between them could be invalid
    # use "erode" to check if two masks are connected
    invalid_edges = []

    for part1, part2, data in G.edges(data=True):
        # Load the corresponding mask images
        part1_path = os.path.join(dir, f"{part1}.png")
        part2_path = os.path.join(dir, f"{part2}.png")
        part1_mask = cv2.imread(part1_path, cv2.IMREAD_GRAYSCALE) # Image.open?
        part2_mask = cv2.imread(part2_path, cv2.IMREAD_GRAYSCALE)

        # dilate
        kernel = np.ones((5, 5), np.uint8)  # Structuring element for dilation
        dilated_part1 = cv2.dilate(part1_mask, kernel, iterations=3)
        dilated_part2 = cv2.dilate(part2_mask, kernel, iterations=3)

        # Check if the dilated masks are connected (bitwise AND to check for overlap)
        overlap = cv2.bitwise_and(dilated_part1, dilated_part2)

        # If there's no overlap (masks are not connected), mark this relation as invalid
        if np.sum(overlap) == 0:
            invalid_edges.append((part1, part2)) # (part1, part2, index) for one edge

    G.remove_edges_from(invalid_edges) # Accessible function in networkx
    return G


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
    # remove "fixed" and change "controllable" to "static"

    invalid_edges = []

    G.remove_edges_from(invalid_edges)
    return G


def detect_redundancy_kr(G, dir):
    # for all
    # A->B->C A->C
    # remove A->C
    invalid_edges = []
    for part1, part2 in G.edges():
        for part3 in G.nodes():
            if part3 != part1 and part3 != part2:
                if G.has_edge(part1, part3) and G.has_edge(part3, part2):
                    invalid_edges.append((part1, part2))
                    break
    G.remove_edges_from(invalid_edges)
    return G
