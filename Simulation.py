import numpy as np
import random
import math

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

from Environment import Environment

NUM_OF_REGIONS = 25
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
    availability = env.bases

def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return true
    return false

def get_accident_location(booleanList):
    for i in booleanList:
        if i == 1:
            accident_index = i
    
    return self.postcode_dic[region][i]

def select_action(state):
    global steps_done
    steps_done += 1
    min_dist_time = 9999
    dist_argmin = 0
    
    for i, base in enumerate(state):
        dist_time = env.distance_time(base, accident_loc)
        if dist_time < min_dist_time:
            min_dist_time = dist_time
            dist_argmin = base

    availability[region][dist_argmin] -= 1
    counter[region][dist_argmin] = env.calculate_ttt(dist_argmin, accident_loc)

    return accident_loc, dist_argmin, min_dist_time, env.calculate_ttt(dist_argmin, accident_loc)

def update_state(action):
    bases_lst = env.bases[region]
    available_bases = []
    for i, base in enumerate(bases_lst):
        if bases_lst[base] - availability[region][base] > 0:
            available_bases.append(int(base))

    return available_bases

def reset(region):
    return env.bases[region]

def run_sim():
    for episode in range(n):
        state = reset()

        for second in range(HOURS*MINUTES*SECONDS):
            for region in range(NUM_OF_REGIONS):
                for counter_ in self.counter[region]:
                    if counter_ > 0:
                        counter_ -= 1
                        if counter_ == 0:
                            self.availability[region][counter_] += 1

                if didAccidentHappen(env.sample_accidents(region)):
                    accident_loc = get_accident_location(env.sample_accidents(region))
                    action = select_action(state)
                    new_state = update_state(action)

