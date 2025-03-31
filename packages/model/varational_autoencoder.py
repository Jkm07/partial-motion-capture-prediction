from packages.model.encoder import Encoder
from packages.model.decoder import Decoder
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
    
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD