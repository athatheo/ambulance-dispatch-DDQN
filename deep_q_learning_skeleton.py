import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EPISODES = 500
MAX_NR_ZIPCODES = 456 # maximum number of zipcodes per region
MAX_NR_AMBULANCE = 18 # maximum number of ambulances per region
DEFAULT_DISCOUNT = 0.99
EPSILON = 1
LEARNINGRATENET = 0.0001  # QNET


class QNet_MLP(nn.Module):
    def __init__(self, num_a, obs_shape, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATENET):
        super(QNet_MLP, self).__init__()

        self.discount = discount
        self.learning_rate = learning_rate

        self.init_network(obs_shape, num_a)
        self.init_optimizer()

    def init_optimizer(self):
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.9)

    def init_network(self, input, num_a):
        """
        Initialization of NN with one 512 hidden layer and masking before output (not sure if this should be applied here)
        :param input: matrix recording where incidents happened and ambulances are available per zip code
        :param num_a: number of available ambulance
        :return:
        """
        num_in = len(input) # number of zip-codes in region
        HIDDEN_NODES1 = 512

        ### MLP
        self.fc1 = nn.Linear(num_in, HIDDEN_NODES1)
        self.fc2 = nn.Linear(HIDDEN_NODES1, MAX_NR_AMBULANCE) #
        self.fc3 = nn.Linear(MAX_NR_AMBULANCE, num_a)  # this should be where masking is applied

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
