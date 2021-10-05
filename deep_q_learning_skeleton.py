import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import debug_utils

################ Define model parameters ################

MAX_NR_ZIPCODES = 456 # maximum number of zipcodes per region
MAX_NR_AMBULANCE = 18 # maximum number of ambulances per region

NUM_EPISODES = 500
# discount rate of future rewards
DEFAULT_DISCOUNT = 0.99
# Exploration rate (epsilon) is probability of choosing a random action
EPSILON = 0.05
# Learning reate for NN
LEARNING_RATE = 0.0001  # QNET
# Save results
RESULTS_NAME = 'ddqn'
# Replay memory size
RMSIZE = 10000

################ Define DQN (Deep Q Network) class ################

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
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.9)

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
        self.fc2 = nn.Linear(HIDDEN_NODES1, MAX_NR_ZIPCODES) #
        #self.fc3 = nn.Linear(MAX_NR_ZIPCODES, num_a)  # this should be where masking is applied

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        """ Defines forward pass through the network on input data x (assumes x to be a tensor) """
        debug_utils.assert_isinstance(x, torch.Tensor)

        x = F.ReLU(self.fc1(x))
        x = F.ReLU(self.fc2(x))
        #x = self.fc3(x)
        return x

    def act(self, state):
        """
        Act either randomly or by predicting action that returns max Q
        :param state: KxN matrix
        """
        if np.random.rand() < self.EPSILON:
            action = random.randrange(self.act_space) # change to choose one of the bases
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.net(torch.FloatTensor(state))
            # Get indec of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        return action


################ Define DQN training class ################

class QLearner(object):

    def __init__(self, env, q_function, qn_target, discount=DEFAULT_DISCOUNT, rm_size=RMSIZE):
        """
        Construct the Q-learner.
        :param env: instance of Environment.py
        :param q_function: instance of Deep Q-network class
        :param qn_target:
        :param discount:
        :param rm_size:
        """
        self.env = env
        self.Q = q_function # this is the dqn
        #self.rm = ReplayMemory(rm_size)  # replay memory stores (a subset of) experience across episode
        self.discount = discount

        self.epsilon = EPSILON
        self.epsilon_min = .01
        self.epsilon_decay = .98

        #self.batch_size = BATCH_SIZE
        self.target_network = qn_target

        self.name = "agent1"
        self.episode = 0
        self.cum_r = 0  # cumulative reward in current episode
        self.tot_r = 0  # cumulative reward in lifetime
        self.stage = 0  # the time step, or 'stage' in this episode
        self.tot_stages = 0  # total time steps in lifetime

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.env.action_space - 1)
        else:
            action = self.Q.argmax_Q_value(state)
        return action

    def process_action(self, action, reward):
        #calculate reward
        return None
