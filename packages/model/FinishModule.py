import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.utils.slices import ROTATION_FLAT

class FinishModule(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_list):
        super(FinishModule, self).__init__()

        self.nodes_convolutions = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size=1) for _ in adjacency_list])

    
    def forward(self, x):
        out = []
        x = x.permute(0, 3, 2, 1)
        for node_idx, node_conv in enumerate(self.nodes_convolutions):
            out.append(node_conv(x[..., node_idx, :]).unsqueeze(2))
        out = torch.cat(out, -2)

        out[ROTATION_FLAT] = F.tanh(out[ROTATION_FLAT])

        return out.permute(0, 3, 2, 1)