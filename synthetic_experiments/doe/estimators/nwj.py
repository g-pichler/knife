import torch.nn as nn
import torch
import numpy as np
from .utils import FF
import math
import torch.nn.functional as F


class NWJ(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(NWJ, self).__init__()
        self.net = FF(args, zc_dim + zd_dim, zc_dim, 1)
        self.measure = args.nwj_measure

    def forward(self, z_c, z_d):
        # shuffle and concatenate
        sample_size = z_d.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()
        z_d_shuffle = z_d[random_index]

        T0 = self.net(torch.cat([z_c, z_d], dim=-1))
        T1 = self.net(torch.cat([z_c, z_d_shuffle], dim=-1))

        positive = get_positive_expectation(T0, self.measure)
        negative = get_negative_expectation(T1, self.measure)

        # lower_bound = torch.abs(T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean())
        lower_bound = positive.mean() - negative.mean()
        return lower_bound, 0., 0.

    def learning_loss(self, z_c, z_d):
        return -self(z_c, z_d)[0]


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise ValueError("Measure incorrect")

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = q_samples.logsumexp(0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise ValueError("Measure incorrect")

    if average:
        return Eq.mean()
    else:
        return Eq