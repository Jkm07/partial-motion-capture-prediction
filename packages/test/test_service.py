
import torch.nn as nn
import torch
import numpy as np
from packages.math import math_utils
from packages.utils import joint_utils

class TestService:
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.test_history = []

    def run_test(self, disable_joints = []):
        mse_list = []
        poss_loss_list = []
        rot_loss_list = []
        with torch.no_grad():
            self.model.eval()
            for data in self.dataloader:
                output, _, _ = self.model(joint_utils.get_data_disable_joint(data, disable_joints) if disable_joints else data)

                mse, poss_loss, rot_loss = self.run_test_for_given_data(output, data, disable_joints)
                mse_list.append(mse)
                poss_loss_list.append(poss_loss)
                rot_loss_list.append(rot_loss)

        test_item = {"mse": np.mean(mse_list), "poss_l2_loss": np.mean(poss_loss_list), "rot_l2q_loss": np.mean(rot_loss_list)}

        self.test_history.append(test_item)
        return test_item
    
    def run_test_for_given_data(self, actual, expected, disable_joints = []):
        mse = 0
        if disable_joints:
            mse = nn.MSELoss()(actual[..., disable_joints, :], expected[..., disable_joints, :])
        else:
            mse = nn.MSELoss()(actual, expected)
        poss_loss, rot_loss = self.get_l2q(actual, expected, disable_joints)
        return mse, poss_loss, rot_loss
    
    def get_idx_of_last_best_result(self, metric = 'mse') -> int:
        array_metric = np.array([m[metric] for m in self.test_history])
        return np.argmin(array_metric)
    
    def is_last_test_improve_result(self, metric = 'mse') -> bool:
        return self.get_idx_of_last_best_result(metric) == len(self.test_history) - 1
    
    def get_l2q(self, actual, expected, disable_joints: list):
        actual_pos = torch.cumsum(actual[..., -1, :3], dim=-3)
        expected_pos = torch.cumsum(expected[..., -1, :3], dim=-3)
        poss_loss = nn.MSELoss()(actual_pos, expected_pos) if actual.shape[-2] -1 not in disable_joints else 0

        return poss_loss, self.get_rot_loss(actual, expected, disable_joints)
    
    def get_rot_loss(self, actual, expected, disable_joints: list):
        disable_joints = [i for i in disable_joints if i != actual.shape[-2] -1]
        actual_rot_quat = math_utils.get_quat_from_matrix(actual[..., :-1, :][..., disable_joints, :] if disable_joints else actual[..., :-1, :])
        expected_rot_quat = math_utils.get_quat_from_matrix(expected[..., :-1, :][..., disable_joints, :] if disable_joints else expected[..., :-1, :])
        return nn.MSELoss()(actual_rot_quat, expected_rot_quat)