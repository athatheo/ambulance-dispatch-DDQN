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
EPISODE_LENGTH = SECONDS * MINUTES * HOURS

def setup_sim():
    env = Environment()
    env.import_data()

def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return True
    return False

def select_action(state):
    min_dist_time = 9999
    min_base = 0
    travel_time = state.travel_time

    for i, trav_time in enumerate(travel_time):
        # dist_time = env.distance_time(base, accident_loc)
        if trav_time == 0:
            continue
        elif trav_time < min_dist_time:
            min_dist_time = trav_time
            min_base = i

    # availability[region_nr][min_base] -= 1
    # counter[region_nr][min_base] = env.calculate_ttt(min_base, accident_loc)

    return min_base
    # return nr_ambulances.index(min_base), min_base, env.calculate_ttt(min_base, accident_loc)

def run_sim():
    env = Environment()
    env.import_data()
    
    for episode in range(NUM_OF_EPISODES):
        region_nr = random.randint(1, NUM_OF_REGIONS+1)
        state = State(env, region_nr)
        tot_reward = 0
        print('------------------')
        print(episode)
        for second in range(EPISODE_LENGTH):
            
            if state.ambulances_return[second]:
                state.nr_ambulances[state.ambulances_return[second]] += 1 
                # if len(state.waiting_list) > 0:
                #     state.nr_ambulances[state.ambulances_return[second]] -= 1
            
            accident_location_list = env.sample_accidents(region_nr)

            if didAccidentHappen(accident_location_list):
                state.update_state(second, accident_location_list)
                action = select_action(state)
                new_state, reward = state.process_action(action, second, accident_location_list)
            else:
                new_state = state
                reward = 0
            tot_reward += reward
            print(second)
            print(reward, tot_reward, action)

            state = new_state

