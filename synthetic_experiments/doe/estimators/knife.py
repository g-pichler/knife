import torch.nn as nn
import torch
import numpy as np
from .utils import FF


class KNIFE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(KNIFE, self).__init__()
        self.kernel_marg = MargKernel(args, zc_dim, zd_dim)
        self.kernel_cond = CondKernel(args, zc_dim, zd_dim)

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        marg_ent = self.kernel_marg(z_d)
        cond_ent = self.kernel_cond(z_c, z_d)
        return marg_ent - cond_ent, marg_ent, cond_ent

    def learning_loss(self, z_c, z_d):
        marg_ent = self.kernel_marg(z_d)
        cond_ent = self.kernel_cond(z_c, z_d)
        return marg_ent + cond_ent

    def I(self, *args, **kwargs):
        return self.forward(*args[:2], **kwargs)[0]


class MargKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim, init_samples=None):

        self.optimize_mu = args.optimize_mu
        self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.d = zc_dim
        self.use_tanh = args.use_tanh
        self.init_std = args.init_std
        super(MargKernel, self).__init__()

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        if init_samples is None:
            init_samples = self.init_std * torch.randn(self.K, self.d)
        # self.means = nn.Parameter(torch.rand(self.K, self.d), requires_grad=True)
        if self.optimize_mu:
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        else:
            self.means = nn.Parameter(init_samples, requires_grad=False)

        if args.cov_diagonal == 'var':
            diag = self.init_std * torch.randn((1, self.K, self.d))
        else:
            diag = self.init_std * torch.randn((1, 1, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)

        if args.cov_off_diagonal == 'var':
            tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
            tri = tri.to(init_samples.dtype)
            self.tri = nn.Parameter(tri, requires_grad=True)
        else:
            self.tri = None

        weigh = torch.ones((1, self.K))
        if args.average == 'var':
            self.weigh = nn.Parameter(weigh, requires_grad=True)
        else:
            self.weigh = nn.Parameter(weigh, requires_grad=False)

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def update_parameters(self, z):
        self.means = z

    def forward(self, x):
        y = -self.logpdf(x)
        return torch.mean(y)


class CondKernel(nn.Module):

    def __init__(self, args, zc_dim, zd_dim, layers=1):
        super(CondKernel, self).__init__()
        self.K, self.d = args.cond_modes, zd_dim
        self.use_tanh = args.use_tanh
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        self.mu = FF(args, self.d, self.d, self.K * self.d)
        self.logvar = FF(args, self.d, self.d, self.K * self.d)

        self.weight = FF(args, self.d, self.d, self.K)
        self.tri = None
        if args.cov_off_diagonal == 'var':
            self.tri = FF(args, self.d, self.d, self.K * self.d * self.d)
        self.zc_dim = zc_dim

    def logpdf(self, z_c, z_d):  # H(X|Y)

        z_d = z_d[:, None, :]  # [N, 1, d]

        w = torch.log_softmax(self.weight(z_c), dim=-1)  # [N, K]
        mu = self.mu(z_c)
        logvar = self.logvar(z_c)
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp().reshape(-1, self.K, self.d)
        mu = mu.reshape(-1, self.K, self.d)
        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")

        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri is not None:
            tri = self.tri(z_c).reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3)
        z = torch.sum(z ** 2, dim=-1)  # [N, K]

        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC.to(z.device) + z

    def forward(self, z_c, z_d):
        z = -self.logpdf(z_c, z_d)
        return torch.mean(z)