
import torch.nn as nn
import torch
import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

class TestService:
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.test_history = []

    def run_test(self):
        self.model.eval()
        mse_list = []
        mase_list = []
        with torch.no_grad():
            for data in self.dataloader:
                output, _, _ = self.model(data)
                mse_list.append(nn.MSELoss()(output, data).cpu().numpy())
                for batch_idx in range(data.size()[0]):
                    y_true = data[batch_idx].cpu().numpy().reshape((data.size()[1], -1))
                    y_pred = output[batch_idx].cpu().numpy().reshape((output.size()[1], -1))
                    y_train = np.zeros_like(y_true[:2])
                    mase_list.append(mean_absolute_scaled_error(y_true, y_pred, y_train = y_train))
        self.model.train()

        test_item = {"mse": np.mean(mse_list), "mase": np.mean(mase_list)}

        self.test_history.append(test_item)
        return test_item
    
    def get_idx_of_last_best_result(self, metric = 'mase') -> int:
        array_metric = np.array([m[metric] for m in self.test_history])
        return np.argmin(array_metric)
    
    def is_last_test_improve_result(self, metric = 'mase') -> bool:
        return self.get_idx_of_last_best_result(metric) == len(self.test_history) - 1