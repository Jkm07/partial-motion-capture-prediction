from packages.model.residual_block import ResidualBlock
import torch.nn as nn
import torch.nn.functional as F
from . import utils

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list, seq_len = 60):
        super(Encoder, self).__init__()

        self.res_block1 = ResidualBlock(in_channels, in_channels, adjacency_list, stride=2)
        self.res_block2 = ResidualBlock(in_channels, in_channels * 2, adjacency_list, stride=2)
        self.res_block3 = ResidualBlock(in_channels * 2, in_channels * 4, adjacency_list, stride=3)

        flat_layer_size = utils.get_flat_layer_size(in_channels, adjacency_list, seq_len)
        self.fc_mu = nn.Linear(flat_layer_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_layer_size, latent_dim)
    
    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar