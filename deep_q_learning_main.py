import copy
import matplotlib
from QNet import QNet_MLP
from AttentionNet import AttentionNet_MLP
from Model import QModel
import numpy as np
from Environment import Environment
from State import State
import shelve
from torch import device, cuda
from Memory import ReplayMemory
from Learner import Learner
import torch
import matplotlib.pyplot as plt

device = device("cuda" if cuda.is_available() else "cpu")

RUN = True
# Method specifies the neural network structure used either "QNet" or "Self-attention"
METHOD = "Self-attention"
SECONDS = 60
MINUTES = 60
HOURS = 24
NUM_EPISODES = 600

NUM_OF_REGIONS = 24
EPISODE_LENGTH = SECONDS * MINUTES * HOURS

EPSILON_MIN = 0.05
EXPLORATION_MAX = 600

rewards_list = [[0] for i in range(25)]
greedy_rewards_list = [[0] for i in range(25)]
difference_list = [[0] for i in range(25)]
max_qvals_list = []

def act_loop(env, agent, replay_memory, learner):
    region_nr = 22
    accidents_happened = env.create_accidents(region_nr)

    for episode in range(NUM_EPISODES):
        #region_nr = 22# np.random.randint(22, 24)#, NUM_OF_REGIONS+1)
        #if region_nr == 13 or region_nr == 14:
            #continue

        state, state_greedy = State(env, region_nr), State(env, region_nr)

        agent.restart_counters(episode)
        agent.update_epsilon(episode, EXPLORATION_MAX)
        counter = 0

        first = True
        current_state_copy = None

        for second in range(EPISODE_LENGTH):
            if second in state.ambulance_return:
                state.nr_ambulances[state.ambulance_return[second]] += 1

                if state.ambulance_return[second] and not(state.ambulance_return[second] in state.indexNotMasked):
                    state.indexNotMasked = np.append(state.indexNotMasked, state.ambulance_return[second])
                state.ambulance_return.pop(second)

            if second in state_greedy.ambulance_return:
                state_greedy.nr_ambulances[state_greedy.ambulance_return.pop(second)] += 1

            if second in accidents_happened:
                counter+=1
                state.update_state(second, accidents_happened[second])
                state_greedy.update_state_greedy(second, accidents_happened[second])

                next_state_copy = copy.deepcopy(state)
                if current_state_copy:
                    replay_memory.push(current_state_copy, action, next_state_copy, torch.tensor([[reward]], device= device))

                if first:
                    first = False
                    qvals = agent.policy_net(state.get_torch())
                    max_qvals_list.append(torch.max(torch.stack([qvals[i] for i in range(len(qvals)) if i in state.indexNotMasked])).item())

                current_state_copy = copy.deepcopy(state)

                action = agent.select_action(state)
                reward = state.process_action(action.item(), second)

                action_greedy = agent.select_action_greedy(state_greedy)
                reward_greedy = state_greedy.process_action_greedy(action_greedy, second)

                agent.cum_r_greedy += reward_greedy
                agent.cum_r += reward
                agent.tot_r += reward
                agent.stage = second
                agent.tot_stages += 1

                if counter % 5 == 0:
                    counter = 0
                    learner.optimize_model(replay_memory)


        rewards_list[region_nr].append(agent.cum_r)
        greedy_rewards_list[region_nr].append(agent.cum_r_greedy)
        difference_list[region_nr].append(agent.cum_r-agent.cum_r_greedy)

        if episode % 10 == 0:
            print("Episode: ", episode + 1)
            agent.update_nets()
            store_data(agent, rewards_list, greedy_rewards_list, difference_list)
        if episode % 100 == 0:
            plot_end(learner.loss_array, max_qvals_list, difference_list)

    print('Complete')
    plot_end(learner.loss_array, max_qvals_list, difference_list)
    return


def plot_end(loss, max_qvals, difference):
    plt.scatter(range(len(loss)), loss)
    plt.title("Loss ")
    plt.show()
    plt.scatter(range(len(max_qvals)), max_qvals)
    plt.title("Max Qvalswt ")
    plt.show()
    plt.scatter(range(len(difference[22])), difference[22])
    plt.title("Difference ML & Greedy")
    plt.show()
    plt.scatter(range(len(greedy_rewards_list[22])), greedy_rewards_list[22])
    plt.title("Greedy")
    plt.show()
    from Visualiser import Visualiser
    vis = Visualiser()
    vis.plot_rolling_average(100, 22)


def store_data(agent, rewards_list, greedy_rewards_list, difference_list):
    model_data = shelve.open('model.txt')
    model_data['model'] = agent
    model_data['rewards'] = rewards_list
    model_data['greedy_rewards_list'] = greedy_rewards_list
    model_data['difference_list'] = difference_list


if RUN:
    environment_data = shelve.open('environment.db.txt')
    env = environment_data['key']
    environment_data.close()
    #env = Environment()
    #env.import_data()

    if METHOD == "QNet":
        policy_net = QNet_MLP(env.state_k).to(device)
        target_net = QNet_MLP(env.state_k).to(device)
    if METHOD == "Self-attention":
        policy_net = AttentionNet_MLP(env.state_k).to(device)
        target_net = AttentionNet_MLP(env.state_k).to(device)

    ql = QModel(env, policy_net, target_net)
    replay_memory = ReplayMemory()
    learner = Learner(ql)
    act_loop(env, ql, replay_memory, learner)
