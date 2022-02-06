import torch.nn as nn
import torch
from .utils import FF
import torch.nn.functional as F


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, args, zc_dim, zd_dim):
        super(CLUBSample, self).__init__()
        self.use_tanh = args.use_tanh
        self.p_mu = FF(args, zc_dim, zc_dim, zd_dim)
        self.p_logvar = FF(args, zc_dim, zc_dim, zd_dim)

    def get_mu_logvar(self, z_c):
        mu = self.p_mu(z_c)
        logvar = self.p_logvar(z_c)
        return mu, logvar

    def loglikeli(self, z_c, z_d):
        mu, logvar = self.get_mu_logvar(z_c)
        return (-(mu - z_d) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, z_c, z_d):
        mu, logvar = self.get_mu_logvar(z_c)

        sample_size = z_c.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - z_d) ** 2 / logvar.exp()
        negative = - (mu - z_d[random_index]) ** 2 / logvar.exp()
        upper_bound = (torch.abs(positive.sum(dim=-1) - negative.sum(dim=-1))).mean()
        return upper_bound / 2., 0., 0.

    def learning_loss(self, z_c, z_d):
        return - self.loglikeli(z_c, z_d)