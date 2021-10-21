import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import debug_utils
import torch.optim as optim
import torch
from torch import device, cuda
# if gpu is to be used
device = device("cuda" if cuda.is_available() else "cpu")
################ Define model parameters ################

MAX_NR_ZIPCODES = 456  # maximum number of zipcodes per region
MAX_NR_AMBULANCE = 18  # maximum number of ambulances per region

NUM_EPISODES = 500
DEFAULT_DISCOUNT = 0.99

# Learning reate for NN
LEARNING_RATE = 0.001  # QNET
# Save results
RESULTS_NAME = 'ddqn'
# Replay memory size
RMSIZE = 10000


################ Define Neural Network ################


class QNet_MLP(nn.Module):
    def __init__(self, num_in, discount=DEFAULT_DISCOUNT, learning_rate=LEARNING_RATE):
        """ Constructor method. Set up NN
        :param act_space: number of actions possible (number of ambulances that can be dispatched
        :param obs_space: number of observation returned for state (number of accidents happening???)
        """

        super(QNet_MLP, self).__init__()

        self.discount = discount
        self.learning_rate = learning_rate

        self.init_network(num_in)
        self.init_optimizer()

    def init_optimizer(self):
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.9)

    def init_network(self, num_in):
        """
        Initialization of NN with one 512 hidden layer and masking before output (not sure if this should be applied here)
        :param input: matrix recording where incidents happened and ambulances are available per zip code
        :param num_a: number of available ambulance (int)
        :return:
        """
        HIDDEN_NODES1 = 512

        ### MLP
        self.fc1 = nn.Linear(num_in, HIDDEN_NODES1)
        self.fc2 = nn.Linear(HIDDEN_NODES1, HIDDEN_NODES1)
        self.fc3 = nn.Linear(HIDDEN_NODES1, 1)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        """ Defines forward pass through the network on input data x (assumes x to be a tensor) """
        x = x.to(device)

        if not isinstance(x, torch.Tensor):
            raise ValueError("Input not a Tensor")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
