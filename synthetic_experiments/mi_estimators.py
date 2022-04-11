import numpy as np
import math

import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tensorboardX import SummaryWriter

sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4, 'lines.markersize': 10})


def control_weights(model):
    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and True:
            torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    model.apply(init_weights)


class FF_DOE(nn.Module):

    def __init__(self, dim_input, dim_output, dropout_rate=0):
        super(FF_DOE, self).__init__()
        self.residual_connection = False
        self.num_layers = 1
        self.layer_norm = True
        self.activation = 'tanh'
        self.stack = nn.ModuleList()
        for l in range(self.num_layers):
            layer = []

            if self.layer_norm:
                layer.append(nn.LayerNorm(dim_input))

            layer.append(nn.Linear(dim_input, dim_output))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[self.activation])
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_output, dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return self.out(x)


class ConditionalPDF(nn.Module):

    def __init__(self, dim, hidden, pdf):
        super(ConditionalPDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.X2Y = FF_DOE(dim, 2 * dim, 0.2)

    def forward(self, Y, X):
        mu, ln_var = torch.split(self.X2Y(X), self.dim, dim=1)
        cross_entropy = compute_negative_ln_prob(Y, mu, ln_var, self.pdf)
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


class PDF(nn.Module):

    def __init__(self, dim, pdf='gauss'):
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


class DOE(nn.Module):

    def __init__(self, dim, hidden, pdf='gauss'):
        super(DOE, self).__init__()
        self.qY = PDF(dim, pdf)
        self.qY_X = ConditionalPDF(dim, hidden, pdf)

    def learning_loss(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        return loss

    def forward(self, X, Y):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        return hY - hY_X


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (torch.abs(positive.sum(dim=-1) - negative.sum(dim=-1))).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

    
    
class TUBA(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(TUBA, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.baseline = nn.Linear(y_dim,1) 

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        log_scores = self.baseline(y_samples)
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # shape [sample_size, sample_size, 1]

        lower_bound = 1 +  T0.mean() - log_scores.mean()  - ((T1 - log_scores).logsumexp(dim=1)- np.log(sample_size)).exp().mean() 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, torch.NumberType):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.).sum(dim=-1)  # [nsample]

        mu_1 = mu.unsqueeze(1)  # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        all_probs = (- (y_samples_1 - mu_1) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(
            dim=-1)  # [nsample, nsample]

        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).to(self.device) * (-20.)
        negative = log_sum_exp(all_probs + diag_mask, dim=0) - np.log(batch_size - 1.)  # [nsample]

        return (positive - negative).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class VarUB(nn.Module):  # variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MultiGaussKernelEE(nn.Module):
    def __init__(self, device, number_of_samples, hidden_size,
                 # [K, d] to initialize the kernel :) so K is the number of points :)
                 average='weighted',  # un
                 cov_diagonal='var',  # diagonal of the covariance
                 cov_off_diagonal='var',  # var
                 ):

        self.K, self.d = number_of_samples, hidden_size
        super(MultiGaussKernelEE, self).__init__()
        self.device = device

        # base_samples.requires_grad = False
        # if kernel_positions in ('fixed', 'free'):
        #    self.mean = base_samples[None, :, :].to(self.args.device)
        # else:
        #    self.mean = base_samples[None, None, :, :].to(self.args.device)  # [1, 1, K, d]

        # self.K = K
        # self.d = d
        self.joint = False

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(
            self.device)

        self.means = nn.Parameter(torch.rand(number_of_samples, hidden_size), requires_grad=True).to(
            self.device)
        if cov_diagonal == 'const':
            diag = torch.ones((1, 1, self.d))
        elif cov_diagonal == 'var':
            diag = torch.ones((1, self.K, self.d))
        else:
            assert False, f'Invalid cov_diagonal: {cov_diagonal}'
        self.diag = nn.Parameter(diag.to(self.device))

        if cov_off_diagonal == 'var':
            tri = torch.zeros((1, self.K, self.d, self.d))
            self.tri = nn.Parameter(tri.to(self.device))
        elif cov_off_diagonal == 'zero':
            self.tri = None
        else:
            assert False, f'Invalid cov_off_diagonal: {cov_off_diagonal}'

        self.weigh = torch.ones((1, self.K), requires_grad=False).to(self.device)
        if average == 'weighted':
            self.weigh = nn.Parameter(self.weigh, requires_grad=True)
        else:
            assert average == 'fixed', f"Invalid average: {average}"

    def logpdf(self, x, y=None):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        if self.tri is not None:
            y = y * self.diag + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        else:
            y = y * self.diag
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(self.diag)), dim=2) + w
        if self.joint:
            y = torch.log(torch.sum(torch.exp(y), dim=-1) / 2)
        else:
            y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def learning_loss(self, x_samples, y=None):
        return -self.forward(x_samples)

    def update_parameters(self, kernel_dict):
        tri = []
        means = []
        weigh = []
        diag = []
        for key, value in kernel_dict.items():  # detach and clone
            tri.append(copy.deepcopy(value.tri.detach().clone()))
            means.append(copy.deepcopy(value.means.detach().clone()))
            weigh.append(copy.deepcopy(value.weigh.detach().clone()))
            diag.append(copy.deepcopy(value.diag.detach().clone()))

        self.tri = nn.Parameter(torch.cat(tri, dim=1), requires_grad=True).to(self.device)
        self.means = nn.Parameter(torch.cat(means, dim=0), requires_grad=True).to(self.device)
        self.weigh = nn.Parameter(torch.cat(weigh, dim=-1), requires_grad=True).to(self.device)
        self.diag = nn.Parameter(torch.cat(diag, dim=1), requires_grad=True).to(self.device)

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def forward(self, x, y=None):
        y = torch.abs(-self.logpdf(x))
        return torch.mean(y)


class FF(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='relu', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        super(FF, self).__init__()
        assert (not residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = residual_connection

        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if l == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)


class MultiGaussKernelCondEE(nn.Module):

    def __init__(self, device,
                 number_of_samples,  # [K, d]
                 x_size, y_size,
                 layers=1,
                 ):
        super(MultiGaussKernelCondEE, self).__init__()
        self.K, self.d = number_of_samples, y_size
        self.device = device

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(self.device)

        # mean_weight = 10 * (2 * torch.eye(K) - torch.ones((K, K)))
        # mean_weight = _c(mean_weight[None, :, :, None])  # [1, K, K, 1]
        # self.mean_weight = nn.Parameter(mean_weight, requires_grad=True)

        self.std = FF(self.d, self.d * 2, self.K, layers)
        self.weight = FF(self.d, self.d * 2, self.K, layers)
        # self.mean_weight = FF(d, hidden, K**2, layers)
        self.mean_weight = FF(self.d, self.d * 2, self.K * x_size, layers)
        self.x_size = x_size

    def _get_mean(self, y):
        # mean_weight = self.mean_weight(y).reshape((-1, self.K, self.K, 1))  # [N, K, K, 1]
        # means = torch.sum(torch.softmax(mean_weight, dim=2) * self.base_X, dim=2)  #[1, K, d]
        means = self.mean_weight(y).reshape((-1, self.K, self.x_size))  # [N, K, d]
        return means

    def logpdf(self, x, y):  # H(X|Y)
        # for data in (x, y):
        # assert len(data.shape) == 2 and data.shape[1] == self.d, 'x has to have shape [N, d]'
        # assert x.shape == y.shape, "x and y need to have the same shape"

        x = x[:, None, :]  # [N, 1, d]

        w = torch.log_softmax(self.weight(y), dim=-1)  # [N, K]
        std = self.std(y).exp()  # [N, K]
        # std = self.std(y)  # [N, K]
        mu = self._get_mean(y)  # [1, K, d]

        y = x - mu  # [N, K, d]
        y = std ** 2 * torch.sum(y ** 2, dim=2)  # [N, K]

        y = -y / 2 + self.d * torch.log(torch.abs(std)) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def pdf(self, x, y):
        return torch.exp(self.logpdf(x, y))

    def forward(self, x, y):
        z = -self.logpdf(x, y)
        return torch.mean(z)


class MIKernelEstimator(nn.Module):
    def __init__(self, device, number_of_samples, x_size, y_size,
                 # [K, d] to initialize the kernel :) so K is the number of points :)
                 average='fixed',  # un
                 cov_diagonal='var',  # diagonal of the covariance
                 cov_off_diagonal='var',  # var
                 use_joint=False):
        super(MIKernelEstimator, self).__init__()
        self.use_joint = use_joint
        self.count = 0
        self.count_learning = 0
        self.kernel_1 = MultiGaussKernelEE(device, number_of_samples, x_size)
        if self.use_joint:
            self.kernel_2 = MultiGaussKernelEE(device, number_of_samples, y_size)
            self.kernel_joint = MultiGaussKernelEE(device, number_of_samples, x_size + y_size)
        else:
            self.kernel_conditional = MultiGaussKernelCondEE(device, number_of_samples, x_size, y_size)
        # self.kernel_joint.joint = True

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        hz_1 = self.kernel_1(x_samples)

        # means = copy.deepcopy(self.kernel_joint.means.data.tolist())
        # self.kernel_joint.update_parameters(
        #    {1: self.kernel_1, 2: self.kernel_2})

        # assert means != copy.deepcopy(self.kernel_joint.means.data.tolist())
        if self.use_joint:
            hz_2 = self.kernel_2(y_samples)
            hz = self.kernel_joint(torch.cat([x_samples, y_samples], dim=1))
            self.count += 1
            return torch.abs(hz_1 + hz_2 - hz)  # abs to stay +
        else:
            hz_g1 = self.kernel_conditional(x_samples, y_samples)
            self.count += 1
            return torch.abs(hz_1 - hz_g1)

    def learning_loss(self, x_samples, y_samples):
        hz_1 = self.kernel_1(x_samples)
        if self.use_joint:
            hz_2 = self.kernel_2(y_samples)
            hz = self.kernel_joint(torch.cat([x_samples, y_samples], dim=1))
            self.count_learning += 1
            return hz_1 + hz_2 + hz
        else:
            hz_g1 = self.kernel_conditional(x_samples, y_samples)
            self.count_learning += 1
            return hz_1 + hz_g1


import torch.nn as nn
import torch
import numpy as np
from torch.distributions import MultivariateNormal

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False, cubic=False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0, 0]
    cov = [[1.0, rho], [rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if cubic:
        y = y ** 3

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        # x = torch.cat([x, torch.randn_like(x).cuda() * 0.3], dim=-1)
        y = torch.from_numpy(y).float().cuda()
    return x, y


def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho ** 2)
    return result


def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result


import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cubic=True):
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    for seed in [1, 2 ,3, 4, 5, 6, 7, 8]:
        set_seed(seed)
        lambda_ = 2

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        suffix = '9.07_{}_{}_{}'.format(cubic, lambda_, seed)

        sample_dim = 20
        batch_size = 128
        hidden_size = 15
        learning_rate = 0.001
        training_steps = 5000
        model_list = ["TUBA", "KNIFE"]  # CLUBSample

        mi_list = [2.0, 4.0, 6.0, 8.0, 10.0]  # , 12.0, 14.0, 16.0, 18.0, 20.0]

        total_steps = training_steps * len(mi_list)

        # train MI estimators with samples

        # train MI estimators with samples

        mi_results = dict()
        for model_name in tqdm(model_list, 'Models'):
            if model_name == 'Kernel_F':
                model = MIKernelEstimator(device, sample_dim // 2, sample_dim).to(device)
            elif model_name == 'KNIFE':
                model = MIKernelEstimator(device, batch_size // 6, sample_dim, sample_dim, use_joint=True).to(device)
            elif model_name == 'DOE':
                model = eval(model_name)(sample_dim, sample_dim).to(device)
            else:
                model = eval(model_name)(sample_dim, sample_dim, hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)

            mi_est_values = []

            start_time = time.time()
            for mi_value in tqdm(mi_list, 'MI'):
                rho = mi_to_rho(mi_value, sample_dim)

                for step in tqdm(range(training_steps), 'Training Loop', position=0, leave=True):
                    batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size=batch_size,
                                                                  to_cuda=torch.cuda.is_available(), cubic=cubic)
                    batch_x = torch.tensor(batch_x).float().to(device)
                    batch_y = torch.tensor(batch_y).float().to(device)
                    model.eval()
                    mi_est_values.append(model(batch_x, batch_y).item())

                    model.train()

                    model_loss = model.learning_loss(batch_x, batch_y)

                    optimizer.zero_grad()
                    model_loss.backward()
                    optimizer.step()

                    del batch_x, batch_y
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print("finish training for %s with true MI value = %f" % (model.__class__.__name__, mi_value))
                # torch.save(model.state_dict(), "./model/%s_%d.pt" % (model.__class__.__name__, int(mi_value)))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            end_time = time.time()
            time_cost = end_time - start_time
            print("model %s average time cost is %f s" % (model_name, time_cost / total_steps))
            mi_results[model_name] = mi_est_values

        import seaborn as sns
        import pandas as pd

        colors = sns.color_palette()

        EMA_SPAN = 200

        ncols = len(model_list)
        nrows = 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
        axs = np.ravel(axs)

        xaxis = np.array(list(range(total_steps)))
        yaxis_mi = np.repeat(mi_list, training_steps)

        for i, model_name in enumerate(model_list):
            plt.sca(axs[i])
            p1 = plt.plot(mi_results[model_name], alpha=0.4, color=colors[0])[0]  # color = 5 or 0
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=4)
            mis_smooth = pd.Series(mi_results[model_name]).ewm(span=EMA_SPAN).mean()

            if i == 0:
                plt.plot(mis_smooth, c=p1.get_color(), label='$\\hat{I}$')
                plt.plot(yaxis_mi, color='k', label='True')
                plt.xlabel('Steps', fontsize=25)
                plt.ylabel('MI', fontsize=25)
                plt.legend(loc='upper left', prop={'size': 15})
            else:
                plt.plot(mis_smooth, c=p1.get_color())
                plt.yticks([])
                plt.plot(yaxis_mi, color='k')
                plt.xlabel('Steps', fontsize=25)

            plt.ylim(0, 15.5)
            plt.xlim(0, total_steps)
            plt.title(model_name, fontsize=35)
            import matplotlib.ticker as ticker

            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            plt.xticks(horizontalalignment="right")
            # plt.subplots_adjust( )

        plt.gcf().tight_layout()
        plt.savefig('mi_est_Gaussian_{}_copy.pdf'.format(suffix), bbox_inches=None)
        # plt.show()

        print('Second part')

        bias_dict = dict()
        var_dict = dict()
        mse_dict = dict()
        for i, model_name in tqdm(enumerate(model_list)):
            bias_list = []
            var_list = []
            mse_list = []
            for j in range(len(mi_list)):
                mi_est_values = mi_results[model_name][training_steps * (j + 1) - 500:training_steps * (j + 1)]
                est_mean = np.mean(mi_est_values)
                bias_list.append(np.abs(mi_list[j] - est_mean))
                var_list.append(np.var(mi_est_values))
                mse_list.append(bias_list[j] ** 2 + var_list[j])
            bias_dict[model_name] = bias_list
            var_dict[model_name] = var_list
            mse_dict[model_name] = mse_list

        # %%

        plt.style.use('default')  # ('seaborn-notebook')

        colors = list(plt.rcParams['axes.prop_cycle'])
        col_idx = [2, 4, 5, 1, 3, 0, 6, 7]

        ncols = 1
        nrows = 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3. * nrows))
        axs = np.ravel(axs)

        for i, model_name in enumerate(model_list):
            plt.sca(axs[0])
            plt.plot(mi_list, bias_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

            plt.sca(axs[1])
            plt.plot(mi_list, var_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

            plt.sca(axs[2])
            plt.plot(mi_list, mse_dict[model_name], label=model_name, marker='d', color=colors[col_idx[i]]["color"])

        ylabels = ['Bias', 'Variance', 'MSE']
        for i in range(3):
            plt.sca(axs[i])
            plt.ylabel(ylabels[i], fontsize=15)

            if i == 0:
                if cubic:
                    plt.title('Cubic', fontsize=17)
                else:
                    plt.title('Gaussian', fontsize=17)
            if i == 1:
                plt.yscale('log')
            if i == 2:
                plt.legend(loc='upper left', prop={'size': 12})
                plt.xlabel('MI Values', fontsize=15)

        plt.gcf().tight_layout()
        plt.savefig('bias_variance_Gaussian_{}.pdf'.format(suffix), bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    main()
