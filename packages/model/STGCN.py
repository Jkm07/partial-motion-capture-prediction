import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from packages.utils.subgroups import SUBGROUP_NODES
import torch.nn.functional as F
from packages.bvhConverter.node import adjacency_list_to_edge_format

class STGCN(nn.Module):
    def __init__(self, single_in_channels, single_out_channels, adjacency_list, stride=1, is_transpose=False):
        super(STGCN, self).__init__()
        self.adjacency_list = adjacency_list

        self.adjacency_list_edge = torch.tensor(adjacency_list_to_edge_format(adjacency_list)).cuda()

        raw_convoltional_nodes = {SUBGROUP_NODES[0]: self._create_convoltion_node(single_in_channels, single_out_channels)}
        self.convoltional_nodes = nn.ModuleDict(raw_convoltional_nodes)

        raw_time_convolutions = {SUBGROUP_NODES[0]: self._create_time_convolotion(single_out_channels, single_out_channels, stride, is_transpose= is_transpose)}
        self.time_convolutions = nn.ModuleDict(raw_time_convolutions)
            

    def _create_convoltion_node(self, single_in_channels, single_out_channels):
        return GCNConv(single_in_channels, single_out_channels)
    
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

        return self.forward_node(x, 0)
    
    def forward_node(self, x_neigbours, node_idx):
        B, T, N, C = x_neigbours.size()
        r = x_neigbours.reshape(B * T, N, C)
        r = self.convoltional_nodes[SUBGROUP_NODES[node_idx]](r, edge_index = self.adjacency_list_edge)
        r = F.relu(r)
        C = r.size()[-1]
        r = r.view(B, T, N, -1).permute(0, 2, 3, 1).reshape(B * N, C, T)
        r = self.time_convolutions[SUBGROUP_NODES[node_idx]](r)
        T = r.size()[-1]
        return r.view(B, N, C, T).permute(0, 3, 1, 2)
