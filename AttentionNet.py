import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import device, cuda

device = device("cuda" if cuda.is_available() else "cpu")

HIDDEN_NODES = 128

class AttentionNet_MLP(nn.Module):
    def __init__(self, num_in, num_head = 8, num_layers = 2, dim_feedforward = HIDDEN_NODES, p_dropout = 0.1):
        """ Constructor method. Set up NN
        :param num_in: number of zip codes in the region
        """

        super(AttentionNet_MLP, self).__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=HIDDEN_NODES, nhead=num_head, dim_feedforward=dim_feedforward, dropout=p_dropout)

        self.net = nn.Sequential(
            nn.Linear(num_in, HIDDEN_NODES),
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=torch.nn.LayerNorm(HIDDEN_NODES)),
            nn.Linear(HIDDEN_NODES, HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(HIDDEN_NODES, 1)
        )


    def forward(self, x):
        """ Defines forward pass through the network on input data x (assumes x to be a tensor) """
        x = x.to(device)
        x = self.net(x.float().unsqueeze(1))
        return F.normalize(x, p=1, dim=0)
