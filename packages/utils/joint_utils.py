import torch

def get_data_disable_joint(data: torch.Tensor, joint_idx) -> torch.Tensor:
    nw_data = data.clone()
    nw_data[..., joint_idx, :] = torch.Tensor([0] * 6).to('cuda')
    return nw_data

def input_dropout(data: torch.Tensor, dropout: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    disable_joint_indexes = torch.randint(0, data.shape[-2], (int(data.shape[-2] * dropout),))
    return get_data_disable_joint(data, disable_joint_indexes), disable_joint_indexes
