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
    def __init__(self, num_in, num_out, discount=DEFAULT_DISCOUNT, learning_rate=LEARNING_RATE):
        """ Constructor method. Set up NN
        :param act_space: number of actions possible (number of ambulances that can be dispatched
        :param obs_space: number of observation returned for state (number of accidents happening???)
        """

        super(QNet_MLP, self).__init__()

        self.discount = discount
        self.learning_rate = learning_rate

        self.init_network(num_in, num_out)
        self.init_optimizer()

    def init_optimizer(self):
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.9)

    def init_network(self, num_in, num_out):
        """
        Initialization of NN with one 512 hidden layer and masking before output (not sure if this should be applied here)
        :param input: matrix recording where incidents happened and ambulances are available per zip code
        :param num_a: number of available ambulance (int)
        :return:
        """
        HIDDEN_NODES1 = 512

        ### MLP
        self.fc1 = nn.Linear(num_in, HIDDEN_NODES1)
        self.fc2 = nn.Linear(HIDDEN_NODES1, num_out)
        self.fc3 = nn.Linear(num_out, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """ Defines forward pass through the network on input data x (assumes x to be a tensor) """
        debug_utils.assert_isinstance(x, torch.Tensor)

        x = F.ReLU(self.fc1(x))
        x = F.ReLU(self.fc2(x))
        x = F.ReLU(self.fc3(x))
        return x

    def act(self, state):
        """
        Has a tensor KxN as an input state and runs the model with it, returning the desired action
        :param state: torch.tensor KxN matrix
        """

        q_values = self.net(torch.FloatTensor(state))
        action = np.argmax(q_values.detach().numpy()[0])

        return action


################ Define DQN training class ################

class QLearner(object):

    def __init__(self, env, policy_net, target_net, discount=DEFAULT_DISCOUNT, rm_size=RMSIZE):
        """
        Construct the Q-learner.
        :param env: instance of Environment.py
        :param policy_net: instance of Deep Q-network class
        :param target_net:
        :param discount:
        :param rm_size:
        """
        self.env = env
        self.policy_net = policy_net # this is the dqn
        #self.rm = ReplayMemory(rm_size)  # replay memory stores (a subset of) experience across episode
        self.discount = discount

        self.epsilon = EPSILON
        self.epsilon_min = .01
        self.epsilon_decay = .98

        #self.batch_size = BATCH_SIZE
        self.target_net = target_net

        self.name = "agent1"
        self.episode = 0
        self.cum_r = 0  # cumulative reward in current episode
        self.tot_r = 0  # cumulative reward in lifetime
        self.stage = 0  # the time step, or 'stage' in this episode
        self.tot_stages = 0  # total time steps in lifetime

    def action_greedy(self, qvals, state):
        """Finds the index of zipcode with max qvalue
        :return index of ambulance base"""

        qvals_selectable = [qvals[i] for i in range(len(qvals)) if i in state.indexNotMasked]
        qvals_selectable = torch.stack(qvals_selectable)
        action = torch.argmax(qvals_selectable)
        action_index = state.IndexesActionsNotMasked[action]

        return action_index

    def action_epsGreedy(self, qvals, state):
        """select action according to epsilon greedy strategy: either random or best according to Qvalue"""
        if random.random() < self.epsilon:
            action_index = random.choice(state.indexNotMasked)
        else:
            action_index = self.action_epsGreedy(qvals, state)
        return action_index

    def training_step(self, state):
        """Trains the policy network and gets the qvalues depending on the state.
        :return action (= index of ambulance to send out),
                qvals (= list with qnet output)
        """

        self.policy_net.train()
        qvals = self.policy_net(state)
        return self.action_epsGreedy(qvals, state), qvals

    def process_action(self, action, reward):
        #calculate reward
        return None