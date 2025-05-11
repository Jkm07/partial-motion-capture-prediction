import torch

def get_data_disable_joint(data: torch.Tensor, joint_idx) -> torch.Tensor:
    nw_data = data.clone()
    nw_data[..., joint_idx, :] = 0
    return nw_data

def input_dropout(data: torch.Tensor, dropout: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    disable_joint_indexes, _ = torch.randperm(data.shape[-2], dtype=torch.int)[:(int(data.shape[-2] * dropout))].sort()
    return get_data_disable_joint(data, disable_joint_indexes), disable_joint_indexes.tolist()
