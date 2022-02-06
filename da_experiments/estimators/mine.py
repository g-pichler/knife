import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FF
import math


class MINE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(MINE, self).__init__()
        self.net = FF(args, zc_dim + zd_dim, zc_dim, 1)

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = z_d.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        z_d_shuffle = z_d[random_index]

        T0 = self.net(torch.cat([z_c, z_d], dim=-1))
        T1 = self.net(torch.cat([z_c, z_d_shuffle], dim=-1))

        mi = T0.mean() - (T1.squeeze().logsumexp(0) - math.log(sample_size))
        return mi, 0., 0.

    def learning_loss(self, z_c, z_d):
        return - self(z_c, z_d)[0]