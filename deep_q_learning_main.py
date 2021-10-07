from deep_q_learning_skeleton import *
from Environment import Environment
from Environment import State
# variable specifying to run training loop or not
RUN = True
SECONDS = 60
MINUTES = 60
HOURS = 24
NUM_EPISODES = 365

NUM_OF_REGIONS = 24
EPISODE_LENGTH = SECONDS * MINUTES * HOURS

def act_loop(env, agent):
    for _ in range(NUM_EPISODES):
        region_nr = np.random.randint(1, NUM_OF_REGIONS+1)
        state = State(env, region_nr)
        for second in range(EPISODE_LENGTH):

            if state.ambulance_return[second]:
                state.nr_ambulances[state.ambulance_return[second]] += 1
                if len(state.waiting_list) > 0:
                    state.nr_ambulances[state.ambulance_return[second]] -= 1

            accident_location_list = env.sample_accidents(region_nr)

            if didAccidentHappen(accident_location_list):
                state.update_state(second, accident_location_list)

                # 2) choose action
                action = agent.select_action(state.get_torch())

                # 3) update action
                agent.process_action(action)

    return None


def didAccidentHappen(booleanList):
    if booleanList.count(1)>0:
        return True
    return False


if RUN:
    # set up environment
    env = Environment()
    env.import_data()

    max_bases_index = max(env.bases, key=lambda x: env.bases[x])
    max_nr_bases = len(env.bases[max_bases_index])

    # set up policy DQN
    policy_net = QNet_MLP(env.state_k, max_nr_bases)

    # set up target DQN
    target_net = QNet_MLP(env.state_k, max_nr_bases)
    # set up Q learner (learning the network weights)
    ql = QLearner(env, policy_net, target_net, DEFAULT_DISCOUNT) # why do we need target_qn?
    act_loop(env, ql)

