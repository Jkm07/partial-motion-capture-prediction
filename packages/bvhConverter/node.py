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

def add_hand_legs_skip_connection(adjacency_list: list):
    LEFT_LEG_IDX = 1
    RIGHT_LEG_IDX = 5
    LEFT_ARM_IDX = 15
    RIGHT_ARM_IDX = 34
    adjacency_list[LEFT_LEG_IDX].append(LEFT_ARM_IDX)
    adjacency_list[LEFT_ARM_IDX].append(LEFT_LEG_IDX)
    adjacency_list[RIGHT_LEG_IDX].append(RIGHT_ARM_IDX)
    adjacency_list[RIGHT_ARM_IDX].append(RIGHT_LEG_IDX)
    return adjacency_list

def adjacency_list_to_edge_format(adjacency_list: list):
    out =[[], []]
    for source, target_neighbours in enumerate(adjacency_list):
        for target in target_neighbours:
            out[0].append(source)
            out[1].append(target)
    return out

