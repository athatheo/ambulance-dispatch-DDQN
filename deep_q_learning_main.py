from deep_q_learning_skeleton import *
from Environment import Environment

# variable specifying to run training loop or not
RUN = True
SECONDS = 60
MINUTES = 60
HOURS = 24

def act_loop(env, agent, num_episodes):
    for episode in range(num_episodes):
        # 1) sample state from environment
        # 2) choose action
        # 3) update action
        return None

if RUN:
    # set up environment
    env = Environment()
    env.import_data()

    num_a = env.action_space # max number of bases available in region
    shape_o = env.state_k # number of parameters passed

    # set up policy DQN
    qn = QNet_MLP(env.state_k)
    # set up target DQN
    target_qn = QNet_MLP(num_a, shape_o)
    # set up Q learner (learning the network weights)
    ql = QLearner(env, qn, target_qn, DEFAULT_DISCOUNT) # why do we need target_qn?

    NUM_EPISODES = SECONDS * MINUTES * HOURS

    act_loop(env, ql, NUM_EPISODES)

