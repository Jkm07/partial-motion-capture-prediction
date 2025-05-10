from packages.model.ResNet import ResNet
from packages.model.FinishModule import FinishModule
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, adjacency_list, seq_len):
        super(Decoder, self).__init__()

        self.adjacency_list = adjacency_list
        self.first_node_size = 424 // (len(adjacency_list)* 2)


        self.fc = nn.Linear(latent_dim, 424)
        self.res_blocks = nn.ModuleList()

        seq_len //= 32
        self.res_blocks.append(ResNet(self.first_node_size, 16, adjacency_list, seq_len))
        seq_len *= 2
        self.res_blocks.append(ResNet(16, 16, adjacency_list, seq_len, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(16, 16, adjacency_list, seq_len))
        seq_len *= 2
        self.res_blocks.append(ResNet(16, 16, adjacency_list, seq_len, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(16, 12, adjacency_list, seq_len,))
        seq_len *= 2
        self.res_blocks.append(ResNet(12, 12, adjacency_list, seq_len, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(12, 12, adjacency_list, seq_len,))
        seq_len *= 2
        self.res_blocks.append(ResNet(12, 12, adjacency_list, seq_len, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(12, 8, adjacency_list, seq_len,))
        seq_len *= 2
        self.res_blocks.append(ResNet(8, 8, adjacency_list, seq_len, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(8, out_channels, adjacency_list, seq_len, with_relu=False))

        self.res_blocks.append(FinishModule(out_channels, out_channels, adjacency_list))
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 2, len(self.adjacency_list), self.first_node_size)
        for block in self.res_blocks:
            x = block(x)
        return x