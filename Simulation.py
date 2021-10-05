import numpy as np
import random
import math

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

from Environment import Environment, State

NUM_OF_REGIONS = 24
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

def get_accident_location(env, region_nr, booleanList):
    for i in booleanList:
        if i == 1:
            accident_index = i
    
    return env.postcode_dic[region_nr][accident_index]

def select_action(state, env, region_nr, accident_loc):
    min_dist_time = 9999
    min_base = 0
    nr_ambulances = state.nr_ambulances
    travel_time = state.travel_time

    for i, base in enumerate(nr_ambulances):
        # dist_time = env.distance_time(base, accident_loc)
        if travel_time < min_dist_time:
            min_dist_time = travel_time
            min_base = base

    # availability[region_nr][min_base] -= 1
    # counter[region_nr][min_base] = env.calculate_ttt(min_base, accident_loc)

    return min_base, min_dist_time, env.calculate_ttt(min_base, accident_loc)

def run_sim():
    env = Environment()
    env.import_data()
    
    for episode in range(NUM_OF_EPISODES):
        region = random.choice(range(1, NUM_OF_REGIONS+1))
        state = State(env, region)

        for second in range(HOURS*MINUTES*SECONDS):
            if didAccidentHappen(env.sample_accidents(region)):
                accident_loc = get_accident_location(env, region, env.sample_accidents(region))
                action = select_action(state, env, region, accident_loc)
                new_state, reward = state.process_action(action, second)
            else:
                new_state = state
                reward = 0

