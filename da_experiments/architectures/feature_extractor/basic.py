import torch.nn as nn
import torch


class basic(nn.Module):
    def __init__(self, args):
        super(basic, self).__init__()
        self.ef_dim = args.ef_dim
        self.dropout = args.dropout
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.feature_dim = self.ef_dim
        self.block1 = nn.Sequential(self.make_layer(self.ef_dim, 3),
                                    self.make_layer(self.ef_dim),
                                    self.make_layer(self.ef_dim),
                                    )
        self.block2 = nn.Sequential(self.make_layer(self.ef_dim, self.ef_dim),
                                    self.make_layer(self.ef_dim),
                                    self.make_layer(self.ef_dim),
                                    )
        self.block3 = nn.Sequential(self.make_layer(self.ef_dim),
                                    self.make_layer(self.ef_dim),
                                    self.make_layer(self.ef_dim),
                                    )
        self.transition_layer = self.make_transition_layer()

    def make_layer(self, out_channels, in_channels=None):
        if in_channels is None:
            in_channels = out_channels
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(out_channels),
                             self.activation)

    def make_transition_layer(self):
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.Dropout(self.dropout))

    def noise(self, x, std=1.0):
        eps = torch.randn(x.size()) * std
        out = x
        if self.training:
            out += eps.to(x.device)
        return out

    def forward(self, x):
        x = self.block1(x)
        x = self.noise(self.transition_layer(x))

        x = self.block2(x)
        x = self.noise(self.transition_layer(x))

        x = self.block3(x)

        x = x.mean(dim=(-2, -1))
        return x