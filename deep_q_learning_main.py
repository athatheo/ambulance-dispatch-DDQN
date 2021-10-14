import copy

from QNet import QNet_MLP
from Model import QModel
import numpy as np
from Environment import Environment
from State import State
import shelve
from torch import device, cuda
from Memory import ReplayMemory
from Learner import Learner
# if gpu is to be used
device = device("cuda" if cuda.is_available() else "cpu")
# variable specifying to run training loop or not
RUN = True
SECONDS = 60
MINUTES = 60
HOURS = 24
NUM_EPISODES = 365
MAX_NR_ZIPCODES = 456  # maximum number of zipcodes per region
NUM_OF_REGIONS = 24
EPISODE_LENGTH = SECONDS * MINUTES * HOURS
DEFAULT_DISCOUNT = 0.99
RMSIZE = 30

def act_loop(env, agent, replay_memory):
    learner = Learner(agent)
    for episode in range(NUM_EPISODES):
        region_nr = 20 #np.random.randint(1, NUM_OF_REGIONS+1)
        state = State(env, region_nr)

        print("Episode: ", episode+1)
        print("Region: ", region_nr)

        for second in range(EPISODE_LENGTH):

            if second in state.ambulance_return:
                state.nr_ambulances[state.ambulance_return[second]] += 1
                if len(state.waiting_list) > 0:
                    state.nr_ambulances[state.ambulance_return[second]] -= 1

            accident_location_list = env.sample_accidents(region_nr)
            if didAccidentHappen(accident_location_list):
                #print("Second: ", second + 1)
                #print("Ambulances left: ", state.nr_ambulances)

                state.update_state(second, accident_location_list)

                # 2) choose action
                action = agent.select_action(state)
                current_state_copy = copy.deepcopy(state)

                next_state, reward = state.process_action(action, second)
                next_state_copy = copy.deepcopy(next_state)
                agent.cum_r += reward
                # Store the transition in memory
                replay_memory.push(current_state_copy, action, next_state_copy, reward)
                learner.optimize_model(replay_memory)

        if episode % 10 == 0:
            agent.update_nets()
        print("Reward: ", agent.cum_r)
    print('Complete')

    return None


def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return True
    return False


if RUN:
    # set up environment
    environment_data = shelve.open('environment.db')
    env = environment_data['key']
    environment_data.close()
    #max_bases_index = max(env.bases, key=lambda x: env.bases[x])
    #max_nr_bases = len(env.bases[max_bases_index])

    # set up policy DQN
    policy_net = QNet_MLP(env.state_k, MAX_NR_ZIPCODES).to(device)

    # set up target DQN
    target_net = QNet_MLP(env.state_k, MAX_NR_ZIPCODES).to(device)
    # set up Q learner (learning the network weights)
    ql = QModel(env, policy_net, target_net, DEFAULT_DISCOUNT) # why do we need target_qn?
    replay_memory = ReplayMemory(RMSIZE)

    act_loop(env, ql, replay_memory)
