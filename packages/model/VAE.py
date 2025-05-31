from packages.model.encoder import Encoder
from packages.model.decoder import Decoder
from packages.utils.subgroups import show_subgroup_structure
from packages.bvhConverter.bvh_converter import get_prepared_adjacency_list
from packages.model.ModelConfig import get_config
import torch.nn as nn
import torch
from torchinfo import summary

ROTATION_MATRIX_SIZE = 6
QUTERNION_MATRIX_SIZE = 5

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, adjacency_list)
        self.decoder = Decoder(in_channels, latent_dim, adjacency_list)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar, mid_tensors = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, mid_tensors), mu, logvar
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
def get_vae_model(arguments):
    adjacency_list = get_prepared_adjacency_list()
    input_size = get_config(arguments).GetInputSize()
    vae = VAE(input_size, 
              arguments.latent_dim, 
              adjacency_list)
    print_model_data(vae, input_size, arguments.sequence_length, adjacency_list)
    return vae.cuda()

def print_model_data(vae: VAE, input_size: int, sequence_length: int, adjacency_list: list):
    show_subgroup_structure(adjacency_list)
    print()
    MOCK_BATCH_SIZE = 8
    summary(vae, input_size=(
        MOCK_BATCH_SIZE, 
        sequence_length, 
        len(adjacency_list), 
        input_size), 
        dtypes=[torch.float64], device="cuda", depth=4)