import torch
import torch.nn as nn
import torch.nn.functional as F
from packages.utils.subgroups import SUBGROUP_NODES

class STGCN(nn.Module):
    def __init__(self, single_in_channels, single_out_channels, adjacency_list, stride = 1, is_transpose = False):
        super(STGCN, self).__init__()

        self.adjacency_list = adjacency_list

        raw_convoltional_nodes = {}
        for i, adjacency_list_item in enumerate(adjacency_list):
            if SUBGROUP_NODES[i] not in raw_convoltional_nodes:
                raw_convoltional_nodes[SUBGROUP_NODES[i]] = self._create_convoltion_node(single_in_channels, single_out_channels, adjacency_list_item)
        self.convoltional_nodes = nn.ModuleDict(raw_convoltional_nodes)

        raw_time_convolutions = {}
        for i, _ in enumerate(adjacency_list):
            if SUBGROUP_NODES[i] not in raw_time_convolutions:
                raw_time_convolutions[SUBGROUP_NODES[i]] = self._create_time_convolotion(single_out_channels, single_out_channels, stride, is_transpose= is_transpose)
        self.time_convolutions = nn.ModuleDict(raw_time_convolutions)
            

    def _create_convoltion_node(self, single_in_channels, single_out_channels, adjacency_list_item):
        return nn.Sequential(nn.Linear(in_features=single_in_channels * len(adjacency_list_item), out_features=single_out_channels),  nn.ReLU()
        )
    
    def _create_time_convolotion(self, in_channels, out_channels, stride, is_transpose = False, kernel_size = 3, padding = 1, padding_mode = 'reflect'):
        if is_transpose:
            return nn.ConvTranspose1d(
                in_channels = in_channels, 
                out_channels = out_channels,
                stride= stride, 
                kernel_size = kernel_size,
                padding = padding,
                output_padding= padding
            )
        else:
            return nn.Conv1d(
                in_channels = in_channels, 
                out_channels = out_channels,
                stride= stride, 
                kernel_size = kernel_size,
                padding = padding,
                padding_mode = padding_mode)

    
    def forward(self, x):
        out = []
        for node_idx, adjacency_list_item in enumerate(self.adjacency_list):
            out.append(self.forward_node(x[..., adjacency_list_item, :], node_idx))
        out = torch.cat(out, -2)

        return out
    
    def forward_node(self, x_neigbours, node_idx):
        B, T, N, C = x_neigbours.size()
        r = x_neigbours.reshape(B * T, N * C)
        r = self.convoltional_nodes[SUBGROUP_NODES[node_idx]](r)
        r = r.view(B, T, -1).permute(0, 2, 1)
        return self.time_convolutions[SUBGROUP_NODES[node_idx]](r).permute(0, 2, 1).unsqueeze(2)