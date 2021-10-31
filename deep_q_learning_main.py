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
NUM_EPISODES = 15000

NUM_OF_REGIONS = 24
EPISODE_LENGTH = SECONDS * MINUTES * HOURS

EPSILON_MIN = 0.05
EXPLORATION_MAX = 10000

rewards_list = [[0] for i in range(25)]
greedy_rewards_list = [[0] for i in range(25)]
difference_list = [[0] for i in range(25)]
max_qvals_list = []

def act_loop(env, agent, replay_memory, learner, accidents_happened = None):
    #if accidents_happened is None:
        #accidents_happened = env.create_accidents(region_nr)

    for episode in range(1):
        region_nr = 22#np.random.randint(1, NUM_OF_REGIONS+1)
        if region_nr == 13 or region_nr == 14:
            continue
        accidents_happened = env.create_accidents(region_nr)

        state, state_greedy = State(env, region_nr), State(env, region_nr)

        agent.restart_counters(episode)
        agent.update_epsilon(episode, EXPLORATION_MAX)
        counter = 0
        agent.epsilon = 0
        first = True
        current_state_copy = None
        temp_rewards = []
        temp_greedy_rewards = []
        temp_difference = []
        for second in range(35000000):
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
                temp_rewards.append(reward)
                temp_greedy_rewards.append(reward_greedy)
                temp_difference.append(reward-reward_greedy)
                agent.cum_r += reward
                agent.tot_r += reward
                agent.stage = second
                agent.tot_stages += 1

                if counter % 5 == 0:
                    counter = 0
                    learner.optimize_model(replay_memory)

        replay_memory.push(current_state_copy, action, None, torch.tensor([[reward]], device=device))

        rewards_list[region_nr].append(agent.cum_r)
        greedy_rewards_list[region_nr].append(agent.cum_r_greedy)
        difference_list[region_nr].append(agent.cum_r-agent.cum_r_greedy)
        plot_end(learner.loss_array, max_qvals_list, difference_list, region_nr)
        plt.scatter(range(len(temp_rewards)), temp_rewards, c = 'blue')
        plt.scatter(range(len(temp_greedy_rewards)), temp_greedy_rewards, c='orange')
        plt.plot(temp_difference, c = 'green')
        plt.xlabel("Accident Time Point")
        plt.ylabel("Rewards")
        plt.show()
        if episode % 30 == 0:
            print("Episode: ", episode + 1)
            agent.update_nets()
            store_data(agent, rewards_list, greedy_rewards_list, difference_list, replay_memory, accidents_happened)
        if episode % 100 == 0:
            plot_end(learner.loss_array, max_qvals_list, difference_list, region_nr)

    print('Complete')
    plot_end(learner.loss_array, max_qvals_list, difference_list)
    return


def plot_end(loss, max_qvals, difference, region_nr):
    plt.scatter(range(len(loss)), loss)
    plt.title("Loss ")
    plt.show()
    plt.scatter(range(len(max_qvals)), max_qvals)
    plt.title("Max Qvalswt ")
    plt.xlabel("Episodes")
    plt.ylabel("Q-Values in first run in episode")
    plt.show()
    plt.scatter(range(len(difference[region_nr])), difference[region_nr])
    plt.title("Difference ML & Greedy")
    plt.xlabel("Episodes")
    plt.ylabel("Reward difference")
    plt.show()
    plt.scatter(range(len(greedy_rewards_list[region_nr])), greedy_rewards_list[region_nr])
    plt.title("Greedy")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()
    from Visualiser import Visualiser
    vis = Visualiser()
    vis.plot_rolling_average(100, region_nr)


def store_data(agent, rewards_list, greedy_rewards_list, difference_list, memory, accidents_happened):
    model_data = shelve.open('model.txt')
    model_data['model'] = agent
    model_data['rewards'] = rewards_list
    model_data['greedy_rewards_list'] = greedy_rewards_list
    model_data['difference_list'] = difference_list
    model_data['memory'] = memory
    model_data['accidents'] = accidents_happened


if RUN:
    environment_data = shelve.open('environment.txt')
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
    loaded_model = shelve.open('best_model_5.txt')
    ql = loaded_model['model']
    replay_memory = loaded_model['memory']
    #accidents_happened = loaded_model['accidents']
    loaded_model.close()
    #replay_memory = ReplayMemory()
    learner = Learner(ql)
    act_loop(env, ql, replay_memory, learner, None)
