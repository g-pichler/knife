import torch.nn as nn


class LabelPredictor(nn.Module):
    def __init__(self, args, input_dim, num_classes):
        super(LabelPredictor, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
