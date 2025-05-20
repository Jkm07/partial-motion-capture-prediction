import torch
import torch.nn.functional as F
from packages.math.math_utils import matrix6D_to_9D_torch
from packages.utils.slices import ROTATION_FLAT, POSITION_FLAT
from packages.utils.subgroups import SUBGROUP_NODES
from dataclasses import dataclass

SMOOTH_WEIGHT = 0.05
POS_WEIGHT = 0.01
KLD_WEIGHT = 0.001

ROTATION_MATRIX_SIZE = 6

@dataclass
class LossDetails:
    rotation_loss: float
    position_loss: float
    smooth_rotation_loss: float
    smooth_position_loss: float
    kld: float
    def get_loss(self):
        return self.rotation_loss + self.position_loss + self.smooth_position_loss + self.smooth_position_loss + self.kld

def vae_loss(actual, expected, mu, logvar, disable_joints = []) -> tuple[torch.Tensor, LossDetails]:
    loss_data = get_loss_data_disable(actual, expected, disable_joints) if disable_joints else get_loss_data(actual, expected)
    ROT_LOSS = rot_loss_9D(loss_data.actual_rotation, loss_data.expected_rotation)
    POS_LOSS = position_loss(loss_data.actual_position, loss_data.expected_position) * POS_WEIGHT
    SMOOTH_ROT_LOSS = rotation_smooth_loss(loss_data.actual_rotation_9d, loss_data.expected_rotation_9d) * SMOOTH_WEIGHT
    SMOOTH_POS_LOSS = position_smooth_loss(loss_data.actual_position, loss_data.expected_position) * SMOOTH_WEIGHT
    KLD = kld_loss(mu, logvar) * KLD_WEIGHT
    loss_details = LossDetails(
        rotation_loss=float(ROT_LOSS), 
        position_loss=float(POS_LOSS),
        smooth_rotation_loss=float(SMOOTH_ROT_LOSS),
        smooth_position_loss=float(SMOOTH_POS_LOSS),
        kld=float(KLD))
    return ROT_LOSS + POS_LOSS + KLD + SMOOTH_ROT_LOSS + SMOOTH_POS_LOSS, loss_details


def rot_loss(actual, expected):
    actual = matrix6D_to_9D_torch(actual)
    expected = matrix6D_to_9D_torch(expected)

    return rot_loss_9D(actual, expected)

#TODO: reconsider renaming
def rot_loss_9D(actual, expected):
    return F.l1_loss(actual, expected)

def rotation_smooth_loss(actual, expected):
    if actual.size()[-1] == 3 and actual.size()[-2] == 3:
        return F.l1_loss(_rotation_differ(actual), _rotation_differ(expected))
    else:
        return F.l1_loss(torch.diff(actual, dim=1), torch.diff(expected, dim=1))

def _rotation_differ(matrix):
    return torch.matmul(matrix[:, 1:, ...], matrix[:, :-1, ...].transpose(-1, -2))

def position_loss(actual, expected):
    return F.l1_loss(actual, expected, reduction='mean')

def position_smooth_loss(actual, expected):
    return F.l1_loss(torch.diff(actual, dim=1), torch.diff(expected, dim=1))

def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

@dataclass
class LossRototationMatrixData:
    actual_rotation: torch.Tensor
    expected_rotation: torch.Tensor
    actual_rotation_9d: torch.Tensor
    expected_rotation_9d: torch.Tensor
    actual_position: torch.Tensor
    expected_position: torch.Tensor

def get_loss_data(actual: torch.Tensor, expected: torch.Tensor) -> LossRototationMatrixData:
    return LossRototationMatrixData(
        actual_rotation=actual[ROTATION_FLAT],
        expected_rotation=expected[ROTATION_FLAT],
        actual_rotation_9d=matrix6D_to_9D_torch(actual[ROTATION_FLAT]) if actual.size()[-1] == ROTATION_MATRIX_SIZE else actual[ROTATION_FLAT],
        expected_rotation_9d=matrix6D_to_9D_torch(expected[ROTATION_FLAT]) if expected.size()[-1] == ROTATION_MATRIX_SIZE else expected[ROTATION_FLAT],
        actual_position=actual[POSITION_FLAT], 
        expected_position=expected[POSITION_FLAT])

def get_loss_data_disable(actual: torch.Tensor, expected: torch.Tensor, disable_joints: list[int]) -> LossRototationMatrixData:
    position_node = len(SUBGROUP_NODES.keys()) - 1
    is_position_node_disable = position_node in disable_joints
    disable_joints = [joint for joint in disable_joints if joint != position_node]
    actual_rotation = get_rotation_matrix(actual, disable_joints)
    expected_rotation = get_rotation_matrix(expected, disable_joints)
    return LossRototationMatrixData(
        actual_rotation=actual_rotation,
        expected_rotation=expected_rotation,
        actual_rotation_9d=matrix6D_to_9D_torch(actual_rotation) if actual_rotation.size()[-1] == ROTATION_MATRIX_SIZE else actual_rotation,
        expected_rotation_9d=matrix6D_to_9D_torch(expected_rotation) if expected_rotation.size()[-1] == ROTATION_MATRIX_SIZE else expected_rotation,
        actual_position=get_position_matrix(actual, is_position_node_disable), 
        expected_position=get_position_matrix(expected, is_position_node_disable))

def get_rotation_matrix(mat: torch.Tensor, disable_joints: list[int]):
    if not disable_joints:
        return torch.zeros_like(mat[ROTATION_FLAT])
    disable_slice = (slice(None), slice(None), disable_joints , ...)
    return mat[disable_slice]

def get_position_matrix(mat: torch.Tensor, is_position_node_disable: bool):
    return mat[POSITION_FLAT] if is_position_node_disable else torch.zeros_like(mat[POSITION_FLAT])



