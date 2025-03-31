from packages.model.residual_block import ResidualBlock
import torch.nn as nn
import torch.nn.functional as F
from . import utils


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list, seq_len = 60):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.adjacency_list = adjacency_list
        self.seq_len = seq_len

        self.fc = nn.Linear(latent_dim, utils.get_flat_layer_size(in_channels, adjacency_list, seq_len))
        self.res_block1 = ResidualBlock(in_channels * 4, in_channels * 4, adjacency_list, stride=3, is_transpose=True, kernel_size=4)
        self.res_block2 = ResidualBlock(in_channels * 4, in_channels * 2, adjacency_list, stride=2, is_transpose=True)
        self.res_block3 = ResidualBlock(in_channels * 2, in_channels, adjacency_list, stride=2, is_transpose=True, with_relu=False)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(x.size(0), self.seq_len // 12, len(self.adjacency_list), self.in_channels * 4)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return x