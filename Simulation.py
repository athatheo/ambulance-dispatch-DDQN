import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Environment import environment

NUMBER_OF_EPISODES = 100
GAMMA = 1.00
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

state_space =
action_space =



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        #return result from NN
    else:
        #return random action

for episode in range(n):
    state = update_state(action)
    action = select_action(state)



