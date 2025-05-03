
import torch.nn as nn
import torch
import numpy as np
from packages.utils import joint_utils
from packages.test.metrics import get_position_metrics, get_rotation_metrics
from packages.model.loss import vae_loss
from dataclasses import dataclass
from packages.model.loss import LossDetails

@dataclass
class TestResult:
    loss: LossDetails
    poss_mse: float
    l2lq: float
    l2q: float

def summaraize_test_result(results: list[TestResult]) -> TestResult:
    rotation_loss_list = [i.loss.rotation_loss for i in results]
    postion_loss_list = [i.loss.position_loss for i in results]
    smth_rotation_loss_list = [i.loss.smooth_rotation_loss for i in results]
    smth_position_loss_list = [i.loss.smooth_position_loss for i in results]
    kld_loss_list = [i.loss.kld for i in results]

    loss_detail = LossDetails(
        rotation_loss=np.mean(rotation_loss_list),
        position_loss=np.mean(postion_loss_list),
        smooth_rotation_loss=np.mean(smth_rotation_loss_list),
        smooth_position_loss=np.mean(smth_position_loss_list),
        kld=np.mean(kld_loss_list))
    
    possition_mse = np.mean([i.poss_mse for i in results])
    l2lq = np.mean([i.l2lq for i in results])
    l2q = np.mean([i.l2q for i in results])

    return TestResult(loss = loss_detail, poss_mse=possition_mse, l2lq=l2lq, l2q=l2q)

class TestService:
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        self.model: nn.Module = model
        self.dataloader: torch.utils.data.DataLoader = dataloader
        self.test_history: list[TestResult] = []

    def run_test(self, disable_joints = []):
        results = []
        with torch.no_grad():
            self.model.eval()
            for data in self.dataloader:
                output, mu, logvar = self.model(joint_utils.get_data_disable_joint(data, disable_joints) if disable_joints else data)

                result = self.run_test_for_given_data(output, data, mu, logvar, disable_joints)
                results.append(result)

        test_item = summaraize_test_result(results)

        self.test_history.append(test_item)
        return test_item
    
    def run_test_for_given_data(self, actual, expected, mu, logvar, disable_joints = []):
        _, loss_details = vae_loss(actual, expected, mu, logvar)
        poss_loss = get_position_metrics(actual, expected, disable_joints)
        l2lq, l2q = get_rotation_metrics(actual, expected, disable_joints)
        return TestResult(loss=loss_details,
                          poss_mse=poss_loss,
                          l2lq=l2lq,
                          l2q=l2q)
    
    def get_idx_of_last_best_result(self, skip_epoch = 1) -> int:
        array_metric = np.array([m.loss.get_loss() for m in self.test_history])[::skip_epoch]
        return np.argmin(array_metric)
    
    def is_last_test_improve_result(self, skip_epoch = 1) -> bool:
        return self.get_idx_of_last_best_result(skip_epoch) == len(self.test_history[::skip_epoch]) - 1
    
    