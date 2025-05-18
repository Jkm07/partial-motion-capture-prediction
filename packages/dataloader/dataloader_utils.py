from packages.dataloader.amass_loader import AmassDataloader
from packages.model.ModelConfig import QueternionConfig, RotationMatrixConfig
from torch.utils.data import DataLoader
import torch
import numpy as np

OFFSET_RATIO = 4

def get_amass_dataloader(dataset_path: str, bath_size: int, sequence_length: int, config: QueternionConfig | RotationMatrixConfig, shuffle=True) -> DataLoader:
    dataset = AmassDataloader(dataset_path, config, window_length= sequence_length, offset= sequence_length // OFFSET_RATIO)
    return DataLoader(dataset, batch_size=bath_size, shuffle=shuffle, collate_fn=collate_fn_cuda)

def collate_fn_cuda(batch):
    return torch.from_numpy(np.stack(batch)).cuda()