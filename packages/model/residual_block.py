import torch
import torch.nn as nn
import torch.nn.functional as F

from .restricted_convolutional_block import RestrictedConvolutionalBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_list, stride=1, kernel_size=3, is_transpose=False, with_relu=True):
        super(ResidualBlock, self).__init__()
        self.with_relu = with_relu
        self.conv1 = RestrictedConvolutionalBlock(in_channels, out_channels, adjacency_list, stride, is_transpose= is_transpose, kernel_size = kernel_size)
        self.bn1 = nn.BatchNorm1d(len(adjacency_list) * out_channels)
        self.conv2 = RestrictedConvolutionalBlock(out_channels, out_channels, adjacency_list)
        self.bn2 = nn.BatchNorm1d(len(adjacency_list) * out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = RestrictedConvolutionalBlock(in_channels, out_channels, adjacency_list, stride, is_transpose=is_transpose, kernel_size = kernel_size)
    
    def forward(self, x):
        out = self.conv1(x)
        B, T, N, C = out.size()
        out = self.bn1(out.permute(0, 2, 3, 1).view(B, N * C, T)).view(B, N, C, T).permute(0, 3, 1, 2)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out.permute(0, 2, 3, 1).view(B, N * C, T)).view(B, N, C, T).permute(0, 3, 1, 2)
        out += self.shortcut(x)
        if self.with_relu:
            return F.relu(out)
        return out