from Environment import Environment
import Simulation as Sim
import matplotlib.pyplot as plt
import numpy as np

env = Environment()
env.import_data()

tot_reward = Sim.run_sim()

plt.plot(np.arange(1,1001), tot_reward[:, -1])
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.ylim(-100_000,0)
plt.title("Total reward per episode")
plt.savefig("greedy_totReward")
plt.show()