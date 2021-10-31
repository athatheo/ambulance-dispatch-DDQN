import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import device, cuda, tanh

device = device("cuda" if cuda.is_available() else "cpu")

HIDDEN_NODES1 = 128


class QNet_MLP(nn.Module):
    def __init__(self, num_in):
        """ Constructor method. Set up NN
        :param num_in: number of zip codes in the region
        """

        super(QNet_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_in, HIDDEN_NODES1),
            nn.Mish(),
            nn.Linear(HIDDEN_NODES1, HIDDEN_NODES1),
            nn.Mish(),
            nn.Linear(HIDDEN_NODES1, HIDDEN_NODES1),
            nn.Mish(),
            nn.Linear(HIDDEN_NODES1, HIDDEN_NODES1),
            nn.Mish(),
            nn.Linear(HIDDEN_NODES1, 1)
        )


    def forward(self, x):
        """ Defines forward pass through the network on input data x (assumes x to be a tensor) """
        x = x.to(device)
        x = self.net(x.float())
        x= F.normalize(x, p=1, dim=0)
        return x
