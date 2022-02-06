import torch.nn as nn
from .utils import FF


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            zc_dim, zd_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            z_c, z_d : samples from X and Y, having shape [sample_size, zc_dim/zd_dim]
    '''

    def __init__(self, args, zc_dim, zd_dim):
        super(CLUB, self).__init__()
        self.use_tanh = args.use_tanh
        self.p_mu = FF(args, zc_dim, zc_dim, zd_dim)
        self.p_logvar = FF(args, zc_dim, zc_dim, zd_dim)
        # self.p_logvar = nn.Sequential(nn.Linear(zc_dim, zd_dim),
        #                               nn.ReLU(),
        #                               nn.Linear(zd_dim, zd_dim),
        #                               nn.Tanh())

    def get_mu_logvar(self, z_c):
        mu = self.p_mu(z_c)
        logvar = self.p_logvar(z_c)
        if self.use_tanh:
            logvar = logvar.tanh()
        return mu, logvar

    def forward(self, z_c, z_d):
        mu, logvar = self.get_mu_logvar(z_c)

        # log of conditional probability of positive sample pairs
        positive = - (mu - z_d) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        z_d_1 = z_d.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((z_d_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()
        mi = (positive.sum(-1) - negative.sum(-1)).mean()
        return mi, 0., 0.

    def learning_loss(self, z_c, z_d):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(z_c)
        return -(-(mu - z_d) ** 2 / logvar.exp() - logvar).sum(1).mean(0)

    def I(self, *args, **kwargs):
        return self.forward(*args[:2], **kwargs)[0]
