from deep_q_learning_skeleton import *
from Environment import Environment
from Environment import State
# variable specifying to run training loop or not
RUN = True
SECONDS = 60
MINUTES = 60
HOURS = 24
NUM_EPISODES = 365

NUM_OF_REGIONS = 25

def act_loop(env, agent, steps):
    for _ in range(NUM_EPISODES):
        for _ in range(steps):
            for region in range(NUM_OF_REGIONS):  # Maybe this should be NUM_OF_REGIONS + 1 since they regions are 1,2,3...25 and not 0,1,2...24
                state = State()
                # Should each region have their own series of states? Since the action applied to a specific state won't have anything to do with other regions
                # 1) sample state from environment
                state.bool_accident = env.sample_accidents(region)

                # 2) choose action
                action = agent.select_action(state.get_torch())

                # 3) update action
                agent.process_action(action)

    return None

if RUN:
    # set up environment
    env = Environment()
    env.import_data()

    num_a = env.action_space # max number of bases available in region
    shape_o = env.state_k # number of parameters passed

    # set up policy DQN
    policy_net = QNet_MLP(env.state_k)
    # set up target DQN
    target_net = QNet_MLP(num_a, shape_o)
    # set up Q learner (learning the network weights)
    ql = QLearner(env, policy_net, target_net, DEFAULT_DISCOUNT) # why do we need target_qn?

    steps = SECONDS * MINUTES * HOURS

    act_loop(env, ql, steps)

