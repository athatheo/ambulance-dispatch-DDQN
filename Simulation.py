from io import IncrementalNewlineDecoder
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import shelve

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

from Environment import Environment, State

NUM_OF_REGIONS = 25
NUM_OF_EPISODES = 1000
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

def make_plot(x, y):
    plt.plot(x, y)

def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return True
    return False

def make_booleanList(n, index):
    lst = [0] * n
    lst[index] = 1

    return lst

def select_action(state):
    min_dist_time = 9999
    min_base = None
    travel_time = state.travel_time

    for i, trav_time in enumerate(travel_time):
        if trav_time == 0:
            pass
        elif trav_time < min_dist_time:
            min_dist_time = trav_time
            min_base = i

    return min_base

def run_sim():
    environment_data = shelve.open('environment.txt')
    env = environment_data['key']
    environment_data.close()
    tot_reward = np.zeros((NUM_OF_EPISODES, EPISODE_LENGTH))
    single_reward = np.zeros((NUM_OF_EPISODES, EPISODE_LENGTH))

    for episode in range(NUM_OF_EPISODES):
    #     region_nr = random.randint(1, NUM_OF_REGIONS)
    # for region_nr in range(1, NUM_OF_REGIONS+1):
        region_nr = 22
        if region_nr == 13:
            continue
        print('Epsiode: ', episode+1)
        print('Region: ', region_nr)
        state = State(env, region_nr)

        for second in range(EPISODE_LENGTH):
            if second in state.ambulance_return.keys():
                state.nr_ambulances[state.ambulance_return[second]] += 1 
                if len(state.waiting_list) > 0:   
                    accident_waiting = state.waiting_list.pop(0)
                    accident_waiting_list = list(accident_waiting)
                    base = accident_waiting_list[0]
                    state.update_state(second, make_booleanList(state.N, state.ambulance_return[second]))
                    action = state.ambulance_return[second]                        
                    # new_state, reward = state.process_action(action, accident_waiting[base], make_booleanList(state.N, state.ambulance_return[second]))
                    new_state, reward = state.process_action(action, second, make_booleanList(state.N, state.ambulance_return[second]))
                    state.nr_ambulances[state.ambulance_return[second]] -= 1
            
            accident_location_list = env.sample_accidents(region_nr)

            if didAccidentHappen(accident_location_list):
                state.update_state(second, accident_location_list)
                action = select_action(state)
                new_state, reward = state.process_action(action, second, accident_location_list)
            else:
                new_state = state
                reward = 0
            
            if second == 0:
                tot_reward[episode, second] = reward
                single_reward[episode, second] = reward
            else:
                tot_reward[episode, second] = tot_reward[episode, second-1] + reward
                single_reward[episode, second] = reward
            state = new_state
        
        if len(state.waiting_list) > 0:
            # tot_reward[episode, -1] = -100_000
            print('Waiting list is not empty')
        print('Total reward: ',  tot_reward[episode, -1])
        print('------------------')
    print('Average reward: ', np.mean(tot_reward[:,-1]))

    # plt.plot(np.arange(EPISODE_LENGTH), tot_reward[episode, :])
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Cumulative reward")
    # plt.savefig("greedy_totalReward")
    # plt.show()

    # plt.plot(np.arange(EPISODE_LENGTH), single_reward[episode, :])
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Single reward")
    # plt.savefig("greedy_singleReward")
    # plt.show()

    plt.plot(np.arange(NUM_OF_EPISODES)+1, tot_reward[:, -1])
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Total reward per episode")
    plt.savefig("greedy_totReward")
    plt.show()
