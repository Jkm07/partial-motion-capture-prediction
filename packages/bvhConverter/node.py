import numpy as np

class Node:
    def __init__(self, name: str, type: str, parent, offset, channels):
        self.name = name
        self.type = type
        self.parent = parent
        self.offset = offset
        self.channels = channels
        self.children = []

def get_adjacency_list(root_node: Node, adjacency_list = [], prev_node_idx = None) -> list:
    curr_idx = len(adjacency_list)
    adjacency_list.append([])

    if prev_node_idx is not None:
        adjacency_list[curr_idx].append(prev_node_idx)

    adjacency_list[curr_idx].append(curr_idx)

    for child in root_node.children:
        if child.type == 'End':
            continue
        adjacency_list[curr_idx].append(len(adjacency_list))
        adjacency_list = get_adjacency_list(child, adjacency_list, curr_idx)
    
    return adjacency_list

def add_position_node_to_adjacency_list(adjacency_list: list):
    adjacency_list[0].append(len(adjacency_list))
    adjacency_list.append([0, len(adjacency_list)])
    return adjacency_list

