import torch
import torch.nn as nn
import torch.nn.functional as F

RELATED_NODES = {
    0: 'Hips',
    1: 'UpLeg',
    2: 'Leg',
    3: 'Foot',
    4: 'Toe',
    5: 'UpLeg',
    6: 'Leg',
    7: 'Foot',
    8: 'Toe',
    9: 'Spine',
    10: 'Spine1',
    11: 'Spine2',
    12: 'Neck',
    13: 'Head',
    14: 'Shoulder',
    15: 'Arm',
    16: 'ForeArm',
    17: 'Hand',
    18: 'Finger1',
    19: 'Finger2',
    20: 'Finger3',
    21: 'Finger1',
    22: 'Finger2',
    23: 'Finger3', 
    24: 'Finger1',
    25: 'Finger2',
    26: 'Finger3', 
    27: 'Finger1',
    28: 'Finger2',
    29: 'Finger3', 
    30: 'Finger1',
    31: 'Finger2',
    32: 'Finger3',  
    33: 'Shoulder',
    34: 'Arm',
    35: 'ForeArm',
    36: 'Hand',
    37: 'Finger1',
    38: 'Finger2',
    39: 'Finger3',
    40: 'Finger1',
    41: 'Finger2',
    42: 'Finger3', 
    43: 'Finger1',
    44: 'Finger2',
    45: 'Finger3', 
    46: 'Finger1',
    47: 'Finger2',
    48: 'Finger3', 
    49: 'Finger1',
    50: 'Finger2',
    51: 'Finger3',
    52: 'Position'  
}

class RestrictedConvolutionalBlock(nn.Module):
    def __init__(self, single_in_channels, single_out_channels, adjacency_list, stride = 1, is_transpose = False, kernel_size = 3):
        super(RestrictedConvolutionalBlock, self).__init__()

        self.adjacency_list = adjacency_list

        raw_convoltional_nodes = {}
        for i, adjacency_list_item in enumerate(adjacency_list):
            if RELATED_NODES[i] not in raw_convoltional_nodes:
                raw_convoltional_nodes[RELATED_NODES[i]] = self._create_convoltion_node(single_in_channels, single_out_channels, adjacency_list_item)
        self.convoltional_nodes = nn.ModuleDict(raw_convoltional_nodes)

        raw_time_convolutions = {}
        for i, _ in enumerate(adjacency_list):
            if RELATED_NODES[i] not in raw_time_convolutions:
                raw_time_convolutions[RELATED_NODES[i]] = self._create_time_convolotion(single_out_channels, single_out_channels, stride, is_transpose= is_transpose, kernel_size=kernel_size)
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
        for i, adjacency_list_item in enumerate(self.adjacency_list):
            sliced_tensor = x[..., adjacency_list_item, :]
            B, T, N, C = sliced_tensor.size()
            r = sliced_tensor.reshape(B * T, N * C)
            r = self.convoltional_nodes[RELATED_NODES[i]](r)
            r = r.reshape(B, T, -1).permute(0, 2, 1)
            r = self.time_convolutions[RELATED_NODES[i]](r).permute(0, 2, 1).unsqueeze(2)
            out.append(r)
        out = torch.cat(out, -2)

        return out