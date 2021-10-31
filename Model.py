import random
import numpy as np
import torch
from torch import device, cuda
device = device("cuda" if cuda.is_available() else "cpu")

# Exploration rate (epsilon) is probability of choosing a random action
EPSILON_MAX = 1.00
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95

LEARNING_RATE = 0.1  # QNET

class QModel(object):

    def __init__(self, env, policy_net, target_net, learning_rate = LEARNING_RATE):
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

        self.epsilon = EPSILON_MAX
        self.epsilon_max = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.learning_rate = learning_rate

        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss_fn = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.name = "agent1"
        self.episode = 0
        self.cum_r_greedy = 0
        self.cum_r = 0  # cumulative reward in current episode
        self.tot_r = 0  # cumulative reward in lifetime
        self.stage = 0  # the time step, or 'stage' in this episode
        self.tot_stages = 0  # total time steps in lifetime

    def get_max_q(self, state):
        """Finds the index of zipcode with max qvalue
        :return index of ambulance base"""

        self.policy_net.train()
        qvals = self.policy_net(state.get_torch())
        state.indexNotMasked = np.sort(state.indexNotMasked)
        qvals_selectable = [qvals[i] for i in range(len(qvals)) if i in state.indexNotMasked]
        if len(qvals_selectable) == 0:
            return torch.tensor([[0]], device=device)
        qvals_selectable = torch.stack(qvals_selectable)

        action = torch.argmax(qvals_selectable)
        action_index = state.indexNotMasked[action]
        return torch.tensor([[action_index]], device=device)

    def select_action(self, state):
        """Selects action according to epsilon greedy strategy: either random or best according to Qvalue"""
        x = random.random()
        if x < self.epsilon:
            if len(state.indexNotMasked) == 0:
                return torch.tensor([[0]], device=device)
            return torch.tensor([[state.indexNotMasked[random.randrange(len(state.indexNotMasked))]]], device=device, dtype=torch.long)
        else:
            return self.get_max_q(state)


    def update_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action_greedy(self, state):
        min_dist_time = 99999
        min_base = None
        travel_time = state.travel_time

        for i, trav_time in enumerate(travel_time):
            if trav_time == 99999:
                pass
            elif trav_time < min_dist_time:
                min_dist_time = trav_time
                min_base = i
        if min_base is None:
            min_base = 0
        return min_base

    def restart_counters(self, episode):
        self.cum_r = 0
        self.cum_r_greedy = 0
        self.stage = 0
        self.episode = episode

    def update_epsilon(self, episode, exploration_max):
        if episode > exploration_max:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = (self.epsilon_max - self.epsilon_min) * (
                        exploration_max - episode) / (exploration_max) + self.epsilon_min
