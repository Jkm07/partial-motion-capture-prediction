from packages.model.encoder import Encoder
from packages.model.decoder import Decoder
from packages.math.math_utils import matrix6D_to_9D_torch
import torch.nn as nn
import torch
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list, seq_len = 60):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, adjacency_list, seq_len)
        self.decoder = Decoder(in_channels, latent_dim, adjacency_list, seq_len)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
def vae_loss(actual, expected, mu, logvar):
    BCE_ROT = rot_lost(actual, expected)
    BCE_POS = F.mse_loss(actual[..., -1, :], expected[..., -1, :], reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE_ROT + BCE_POS + KLD, (float(BCE_ROT), float(BCE_POS), float(KLD))

def rot_lost(actual, expected):
    actual = matrix6D_to_9D_torch(actual[..., :-1, :])
    expected = matrix6D_to_9D_torch(expected[..., :-1, :]).transpose(-1, -2)
    ones = torch.eye(3).expand(*expected.shape)

    return F.mse_loss(ones, torch.matmul(actual, expected), reduction='sum')