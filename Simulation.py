import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Environment import Environment

NUM_OF_REGIONS = 24 #region 13 does not exists
NUM_OF_EPISODES = 100
GAMMA = 1.00
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

#state_space =
#action_space =


SECONDS = 60
MINUTES = 60
HOURS = 24

def setup_sim():
    env = Environment()
    env.import_data()

def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return True
    return False

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return 0        #return result from NN
    else:
        return 0#return random action

def run():
    for episode in range(n):
        state = update_state(action)
        action = select_action(state)

        for second in range(HOURS*MINUTES*SECONDS):
            for region in range(NUM_OF_REGIONS):
                if didAccidentHappen(env.sample_accident(region)):
                


