import torch


def generate_latent(latent_dimension, num=1):
    '''
    Generate latent z from a random uniform.
    '''
    return torch.randn(num, latent_dimension, 1, 1)
