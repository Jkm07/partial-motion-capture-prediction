from packages.dataloader.amass_loader import AmassDataloader
from torch.utils.data import DataLoader
import torch
import numpy as np

def get_amass_dataloader(dataset_path: str, bath_size: int, shuffle=True) -> DataLoader:
    dataset = AmassDataloader(dataset_path)
    return DataLoader(dataset, batch_size=bath_size, shuffle=shuffle, collate_fn=collate_fn_cuda)

def collate_fn_cuda(batch):
    return torch.from_numpy(np.stack(batch)).cuda()