import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import FF


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


class SMILE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(SMILE, self).__init__()
        # assert x_dim == y_dim
        self.clip = args.clip
        self.net = FF(args, zc_dim + zd_dim, zc_dim, 1)
        # self.T_func = ConcatCritic(x_dim, hidden_size)

    def forward(self, z_c, z_d): 
        """
        z_c: [N, c]
        z_d: [N, c]
        """
        # f = self.net(torch.cat([z_c, z_d], dim=-1))  # [N]
        # if self.clip is not None:
        #     f_ = torch.clamp(f, -self.clip, self.clip)
        # else:
        #     f_ = f
        sample_size = z_d.shape[0]

        zc_tile = z_c.unsqueeze(0).repeat((sample_size, 1, 1))  # [sample_size, sample_size, c]
        zd_tile = z_d.unsqueeze(1).repeat((1, sample_size, 1))  # [sample_size, sample_size, c]

        T1 = self.net(torch.cat([zc_tile, zd_tile], dim=-1)).squeeze()  # [sample_size, sample_size]

        assert list(T1.size()) == [sample_size, sample_size]

        T0 = T1.diag().mean()

        if self.clip is not None:
            T1 = torch.clamp(T1, -self.clip, self.clip)
        T1 = logmeanexp_nodiag(T1, dim=(0, 1))

        dv = T0 - T1

        # dv = f.diag().mean() - z
        # js = js_fgan_lower_bound(f)
        # with torch.no_grad():
        #     dv_js = dv - js
        return dv, 0., 0.

    def learning_loss(self, z_c, z_d):
        return -self(z_c, z_d)[0]


def logmeanexp_nodiag(x, dim=None, device='cuda'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)
    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)
    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)