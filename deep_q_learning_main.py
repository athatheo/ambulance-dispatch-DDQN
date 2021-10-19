import copy
import matplotlib
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
NUM_EPISODES = 1000
MAX_NR_ZIPCODES = 456  # maximum number of zipcodes per region
NUM_OF_REGIONS = 24
EPISODE_LENGTH = SECONDS * MINUTES * HOURS
DEFAULT_DISCOUNT = 0.99
RMSIZE = 30

rewards_list = [[0] for i in range(25)]

def act_loop(env, agent, replay_memory):
    for episode in range(NUM_EPISODES):
        region_nr = np.random.randint(1, NUM_OF_REGIONS+1)
        if region_nr == 13 or region_nr == 14:
            continue
        state = State(env, region_nr)
        learner = Learner(agent)
        replay_memory = ReplayMemory(RMSIZE)

        print("Episode: ", episode+1)
        print("Region: ", region_nr)
        agent.cum_r = 0
        agent.stage = 0
        #print("Initial ambulances: ", state.nr_ambulances)
        for second in range(EPISODE_LENGTH):

            if second in state.ambulance_return:
                state.nr_ambulances[state.ambulance_return[second]] += 1
                if state.ambulance_return[second] and not(state.ambulance_return[second] in state.indexNotMasked):
                    state.indexNotMasked = np.append(state.indexNotMasked, state.ambulance_return[second])
                if len(state.waiting_list) > 0:
                    # process action
                    # pop from waiting list
                    # maybe sth with reward
                    # maybe new action/state and reward
                    state.nr_ambulances[state.ambulance_return[second]] -= 1

            zip_code_index = env.sample_accidents(region_nr)
            if zip_code_index:
                #print("Second: ", second + 1)
                #print("Ambulances left: ", state.nr_ambulances)

                state.update_state(second, zip_code_index)

                # 2) choose action
                action = agent.select_action(state)
                current_state_copy = copy.deepcopy(state)

                next_state, reward = state.process_action(action, second)
                next_state_copy = copy.deepcopy(next_state)

                agent.cum_r += reward
                agent.tot_r += reward
                agent.stage = second
                agent.tot_stages += 1
                # Store the transition in memory
                replay_memory.push(current_state_copy, action, next_state_copy, reward)
                learner.optimize_model(replay_memory)

        if episode % 10 == 0:
            agent.update_nets()
        print("Reward: ", agent.cum_r)
        print("---------------------------")
        rewards_list[region_nr].append(agent.cum_r)
    print('Complete')
    model_data = shelve.open('model.txt')
    model_data['model'] = agent
    model_data['rewards'] = rewards_list
    return None


def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return True
    return False


if RUN:
    # set up environment
    environment_data = shelve.open('environment.txt')
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
