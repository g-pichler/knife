import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FF
import math


class DoE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(DoE, self).__init__()
        self.qY = PDF(zd_dim, 'gauss')
        self.qY_X = ConditionalPDF(args, zd_dim, zd_dim, 'gauss')

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        hY = self.qY(z_d)
        hY_X = self.qY_X(z_d, z_c)
        mi = hY - hY_X
        return mi, hY, hY_X

    def learning_loss(self, z_c, z_d):
        hY = self.qY(z_d)
        hY_X = self.qY_X(z_d, z_c)
        loss = hY + hY_X
        return loss


class ConditionalPDF(nn.Module):

    def __init__(self, args, dim, hidden, pdf):
        super(ConditionalPDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.X2Y = FF(args, dim, hidden, 2 * dim)

    def forward(self, Y, X):
        mu, ln_var = torch.split(self.X2Y(X), self.dim, dim=1)
        cross_entropy = compute_negative_ln_prob(Y, mu, ln_var, self.pdf)
        return cross_entropy


class PDF(nn.Module):

    def __init__(self, dim, pdf):
        super(PDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.mu = nn.Embedding(1, self.dim)
        self.ln_var = nn.Embedding(1, self.dim)  # ln(s) in logistic

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight,
                                                 self.ln_var.weight, self.pdf)
        return cross_entropy


def compute_negative_ln_prob(Y, mu, ln_var, pdf):
    var = ln_var.exp()

    if pdf == 'gauss':
        negative_ln_prob = 0.5 * ((Y - mu) ** 2 / var).sum(1).mean() + \
                           0.5 * Y.size(1) * math.log(2 * math.pi) + \
                           0.5 * ln_var.sum(1).mean()

    elif pdf == 'logistic':
        whitened = (Y - mu) / var
        adjust = torch.logsumexp(
            torch.stack([torch.zeros(Y.size()).to(Y.device), -whitened]), 0)
        negative_ln_prob = whitened.sum(1).mean() + \
                           2 * adjust.sum(1).mean() + \
                           ln_var.sum(1).mean()

    else:
        raise ValueError('Unknown PDF: %s' % (pdf))

    return negative_ln_prob