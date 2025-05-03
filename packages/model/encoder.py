from packages.model.ResNet import ResNet
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list, seq_len):
        super(Encoder, self).__init__()

        self.res_blocks = nn.ModuleList()

        self.res_blocks.append(ResNet(in_channels, 8, adjacency_list, seq_len))
        seq_len //= 2
        self.res_blocks.append(ResNet(8, 8, adjacency_list, seq_len, stride=2))

        self.res_blocks.append(ResNet(8, 8, adjacency_list, seq_len))
        seq_len //= 2
        self.res_blocks.append(ResNet(8, 12, adjacency_list, seq_len, stride=2))

        self.res_blocks.append(ResNet(12, 12, adjacency_list, seq_len))
        seq_len //= 2
        self.res_blocks.append(ResNet(12, 12, adjacency_list, seq_len, stride=2))

        self.res_blocks.append(ResNet(12, 12, adjacency_list, seq_len))
        seq_len //= 2
        self.res_blocks.append(ResNet(12, 16, adjacency_list, seq_len, stride=2))

        self.res_blocks.append(ResNet(16, 16, adjacency_list, seq_len))
        seq_len //= 2
        self.res_blocks.append(ResNet(16, 16, adjacency_list, seq_len, stride=2))

        self.res_blocks.append(ResNet(16, 16, adjacency_list, seq_len))
        seq_len //= 2

        self.pooling = nn.MaxPool1d(kernel_size=4)

        self.fc_mu = nn.Linear(424, latent_dim)
        self.fc_logvar = nn.Linear(424, latent_dim)
    
    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.pooling(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar