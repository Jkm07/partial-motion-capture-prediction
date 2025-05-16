import torch.nn as nn
import torch.nn.functional as F

from .STGCN import STGCN

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_list, seq_len, stride=1, is_transpose=False, with_relu=True):
        super(ResNet, self).__init__()
        self.with_relu = with_relu
        self.conv1 = STGCN(in_channels, out_channels, adjacency_list, stride, is_transpose= is_transpose)
        self.bn1 = nn.LayerNorm([len(adjacency_list), out_channels])
        self.conv2 = STGCN(out_channels, out_channels, adjacency_list)
        self.bn2 = nn.LayerNorm([len(adjacency_list), out_channels])
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = STGCN(in_channels, out_channels, adjacency_list, stride, is_transpose=is_transpose)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        if self.with_relu:
            return F.relu(out)
        return out