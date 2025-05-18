from packages.model.ResNet import ResNet
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list):
        super(Encoder, self).__init__()

        self.res_blocks = nn.ModuleList()

        self.res_blocks.append(ResNet(in_channels, 32, adjacency_list))
        self.res_blocks.append(ResNet(32, 32, adjacency_list, stride=2))

        self.res_blocks.append(ResNet(32, 32, adjacency_list)) #mid_transfer
        self.res_blocks.append(ResNet(32, 48, adjacency_list, stride=2))

        self.res_blocks.append(ResNet(48, 48, adjacency_list))
        self.res_blocks.append(ResNet(48, 48, adjacency_list, stride=2))

        self.res_blocks.append(ResNet(48, 48, adjacency_list)) #mid_transfer
        self.res_blocks.append(ResNet(48, 64, adjacency_list, stride=2))

        self.res_blocks.append(ResNet(64, 64, adjacency_list))
        self.res_blocks.append(ResNet(64, 64, adjacency_list, stride=2))

        self.res_blocks.append(ResNet(64, 64, adjacency_list)) #mid_transfer

        self.pooling = nn.MaxPool1d(kernel_size=16)

        self.fc_mu = nn.Linear(424, latent_dim)
        self.fc_logvar = nn.Linear(424, latent_dim)
    
    def forward(self, x):
        mid_tensors = []
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if (i - 2) % 4 == 0:
                mid_tensors.append(x)
        x = x.view(x.size(0), -1)
        x = self.pooling(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, mid_tensors