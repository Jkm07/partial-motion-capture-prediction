from packages.model.encoder import Encoder
from packages.model.decoder import Decoder
import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, adjacency_list, seq_len = 65):
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