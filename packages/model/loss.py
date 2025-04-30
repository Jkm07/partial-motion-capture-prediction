import torch
import torch.nn.functional as F
from packages.math.math_utils import matrix6D_to_9D_torch


def vae_loss(actual, expected, mu, logvar):
    BCE_ROT = rot_loss(actual[..., :-1, :], expected[..., :-1, :])
    BCE_POS = F.mse_loss(actual[..., -1, :], expected[..., -1, :], reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE_ROT + BCE_POS + KLD, (float(BCE_ROT), float(BCE_POS), float(KLD))

def rot_loss(actual, expected):
    actual = matrix6D_to_9D_torch(actual)
    expected = matrix6D_to_9D_torch(expected).transpose(-1, -2)
    ones = torch.eye(3).expand(*expected.shape).to('cuda')

    return F.mse_loss(ones, torch.matmul(actual, expected), reduction='sum')