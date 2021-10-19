import random

import torch

# Exploration rate (epsilon) is probability of choosing a random action
EPSILON_MAX = 1.00
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95

# discount rate of future rewards
DEFAULT_DISCOUNT = 0.99

RMSIZE = 100

class QModel(object):

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

        self.epsilon = EPSILON_MAX
        self.epsilon_max = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        #self.batch_size = BATCH_SIZE
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.name = "agent1"
        self.episode = 0
        self.cum_r = 0  # cumulative reward in current episode
        self.tot_r = 0  # cumulative reward in lifetime
        self.stage = 0  # the time step, or 'stage' in this episode
        self.tot_stages = 0  # total time steps in lifetime

    def get_max_q(self, state):
        """Finds the index of zipcode with max qvalue
        :return index of ambulance base"""

        self.policy_net.train()
        qvals = self.policy_net(state.get_torch())
        qvals_selectable = [qvals[i] for i in range(len(qvals)) if i in state.indexNotMasked]
        if len(qvals_selectable) == 0:
            return -1
        qvals_selectable = torch.stack(qvals_selectable)
        action = torch.argmax(qvals_selectable)
        action_index = state.indexNotMasked[action]

        return action_index

    def decrement_epsilon(self):
        return self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
    
    def select_action(self, state):
        """Selects action according to epsilon greedy strategy: either random or best according to Qvalue"""
        self.epsilon = self.decrement_epsilon()
        if random.random() < self.epsilon:
            if len(state.indexNotMasked) == 0:
                return -1
            return random.choice(state.indexNotMasked)
        else:
            return self.get_max_q(state)


    def update_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
