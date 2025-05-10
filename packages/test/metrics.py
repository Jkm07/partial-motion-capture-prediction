import torch.nn as nn
import torch

from packages.math import math_utils
from packages.utils.slices import POSITION_FLAT
from packages.bvhConverter.bvh_converter import get_default_hierarchy
from packages.bvhConverter.node import Node

SKIP_POSITION_UNITS = (..., slice(None, 3))
POSITION_NODE_IDX = 52
    
def get_position_metrics(actual, expected, disable_joints: list[int]):
        if POSITION_NODE_IDX in disable_joints or not disable_joints:
            return get_position_mse(actual, expected)
        else:
            return 0
    
def get_rotation_metrics(actual, expected, disable_joints: list[int]):
    selected_nodes = select_nodes_to_rotation_evaluate(disable_joints)
    l2lq = get_l2lq_from_quternions(actual, expected, selected_nodes)
    l2q = get_l2q_from_quternion(actual, expected, selected_nodes)
    return l2lq, l2q

def select_nodes_to_rotation_evaluate(disable_joints: list[int]):
    disable_joints = [i for i in disable_joints if i != POSITION_NODE_IDX]
    if not disable_joints:
        return (..., slice(None, -1), slice(None)) # remove positional node
    else:
        return (..., disable_joints, slice(None))

def get_position_mse(actual, expected):
        return nn.MSELoss()(
            actual[POSITION_FLAT][SKIP_POSITION_UNITS], 
            expected[POSITION_FLAT][SKIP_POSITION_UNITS]).cpu().numpy()

def get_l2lq_from_quternions(actual, expected, selected_nodes: slice):
    actual_quat = math_utils.from_decompose_quternion(actual[selected_nodes])
    expected_quat = math_utils.from_decompose_quternion(expected[selected_nodes])
    return nn.MSELoss()(actual_quat, expected_quat).cpu().numpy()
    
def get_l2lq_from_rotation(actual, expected, selected_nodes: slice):
    actual_rot_quat = math_utils.get_quat_from_matrix(actual[selected_nodes])
    expected_rot_quat = math_utils.get_quat_from_matrix(expected[selected_nodes])
    return nn.MSELoss()(actual_rot_quat, expected_rot_quat).cpu().numpy()

def get_l2q_from_quternion(actual, expected, selected_nodes: slice):
    actual = prepare_data_for_l2q_quat(actual[selected_nodes])
    expected = prepare_data_for_l2q_quat(expected[selected_nodes])
    return nn.MSELoss()(actual, expected).cpu().numpy()

def get_l2q_from_rotation(actual, expected, selected_nodes: slice):
    actual = prepare_data_for_l2q(actual[selected_nodes])
    expected = prepare_data_for_l2q(expected[selected_nodes])
    return nn.MSELoss()(actual, expected).cpu().numpy()

def prepare_data_for_l2q(matrix):
    matrix = math_utils.matrix6D_to_9D_torch(matrix)
    matrix = get_global_rotations(matrix, lambda x, y: x @ y)
    return math_utils.matrix9D_to_quat_torch(matrix)

def get_global_rotations(rotations, rotate_lambda):
    hierarchy = get_default_hierarchy()
    rotations = rotations.clone()
    rotate_node_with_childs(rotations, torch.tensor([1, 0, 0, 0]).cuda(), 0, hierarchy[0], rotate_lambda)
    return rotations

def rotate_node_with_childs(rotations: torch.Tensor, parent_rotation: torch.Tensor, curr_rot: int, curr_node: Node, rotate_lambda) -> torch.Tensor:
    curr_rot_slice = (..., curr_rot, slice(None)) #diff between quat and rotation matrix
    rotations[curr_rot_slice] = rotate_lambda(rotations[curr_rot_slice], parent_rotation)
    for child in curr_node.children:
        if child.type == 'End':
            continue
        curr_rot = rotate_node_with_childs(rotations, rotations[curr_rot_slice], curr_rot + 1, child, rotate_lambda)
    return curr_rot

def prepare_data_for_l2q_quat(matrix):
    matrix = math_utils.from_decompose_quternion(matrix)
    matrix = get_global_rotations(matrix, math_utils.quaternion_multiply_torch)
    return matrix
          
          