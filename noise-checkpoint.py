import torch
import torch.nn as nn
import torch.optim as optim

from .pl_models import *
from .pl_utils import *
from .mlp import *

def make_noise_dt(A_dt, output_dim, Nλ):
    def noise_dt(x, x_mean):
        """
        Generates the noise with the following input
        x: the input the physical NN
        x_mean: the mean output of the physical NN
        The reason for supplying the x_mean is to avoid recomputing the mean twice!
        Just does A*z where z is iid to give you a sample of noise
        """
        device = x.device

        A_x = torch.cat([x, x_mean], dim=1)
        with torch.no_grad():
            A_y = A_dt(A_x)

        A_mat = A_y.reshape([x.shape[0], output_dim, Nλ])
        z = torch.randn([x.shape[0], Nλ, 1], device=device)
        noise = torch.bmm(A_mat, z)
        return torch.squeeze(noise)
    return noise_dt

def make_dt_func(mean_dt, A_dt, output_dim, Nλ):
    noise_dt = make_noise_dt(A_dt, output_dim, Nλ)
    def f(x):
        x_mean = mean_dt(x)
        noise = noise_dt(x, x_mean)
        return x_mean + noise
    return f