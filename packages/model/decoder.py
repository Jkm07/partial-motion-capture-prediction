from packages.model.ResNet import ResNet
from packages.model.FinishModule import FinishModule
import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, adjacency_list):
        super(Decoder, self).__init__()

        self.adjacency_list = adjacency_list
        self.first_node_size = 424 // (len(adjacency_list)* 2)


        self.fc = nn.Linear(latent_dim, 424)
        self.res_blocks = nn.ModuleList()

        self.res_blocks.append(ResNet(self.first_node_size, 64, adjacency_list))
        self.res_blocks.append(ResNet(128, 64, adjacency_list, stride=2, is_transpose=True))  #mid_transfer

        self.res_blocks.append(ResNet(64, 64, adjacency_list))
        self.res_blocks.append(ResNet(64, 64, adjacency_list, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(64, 48, adjacency_list,))
        self.res_blocks.append(ResNet(96, 48, adjacency_list, stride=2, is_transpose=True))  #mid_transfer

        self.res_blocks.append(ResNet(48, 48, adjacency_list,))
        self.res_blocks.append(ResNet(48, 48, adjacency_list, stride=2, is_transpose=True))

        self.res_blocks.append(ResNet(48, 32, adjacency_list,))
        self.res_blocks.append(ResNet(64, 32, adjacency_list, stride=2, is_transpose=True))  #mid_transfer

        self.res_blocks.append(ResNet(32, 32, adjacency_list))

        self.res_blocks.append(FinishModule(32, out_channels, adjacency_list))
    
    def forward(self, x, mid_tensors):
        mid_tensor_iter = iter(mid_tensors[::-1])
        x = self.fc(x)
        x = x.view(x.size(0), 2, len(self.adjacency_list), self.first_node_size)
        for i, block in enumerate(self.res_blocks):
            if (i - 1) % 4 == 0:
                x = torch.concatenate((x, next(mid_tensor_iter)), dim=-1)
            x = block(x)
        return x