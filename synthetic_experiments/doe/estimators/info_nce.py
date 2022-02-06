import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .utils import FF
import math


class InfoNCE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(InfoNCE, self).__init__()
        # self.net = MINet(args, zc_dim + zd_dim)
        self.net = FF(args, zc_dim + zd_dim, zc_dim, 1)

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = z_d.shape[0]

        zc_tile = z_c.unsqueeze(0).repeat((sample_size, 1, 1))  # [sample_size, sample_size, c]
        zd_tile = z_d.unsqueeze(1).repeat((1, sample_size, 1))  # [sample_size, sample_size, c]

        T0 = self.net(torch.cat([z_c, z_d], dim=-1))
        T1 = self.net(torch.cat([zc_tile, zd_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound, 0., 0.

    def learning_loss(self, z_c, z_d):
        return - self(z_c, z_d)[0]