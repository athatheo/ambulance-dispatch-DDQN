import Environment
import shelve
from State import State
import torch
import numpy as np

env = Environment.Environment()
# env.import_data()
# print(env.nr_ambulances)
# max_key = max(env.nr_ambulances, key= lambda x: env.nr_ambulances[x])
# print(env.nr_ambulances[max_key])
environment_data = shelve.open('environment.txt')
env = environment_data['key']
environment_data.close()

print(env.postcode_dic[1])
print("00000000000000000000000000000")

from torch import cuda

print(cuda.is_available())
state = State(env, 1)
print(env.prob_acc[1])


def foo1():
    accident_prob = env.prob_acc[17]
    random_values = torch.rand(len(accident_prob))
    for zip_code_index in range(len(accident_prob)):
        if random_values[zip_code_index] <= accident_prob[zip_code_index]:  # :
            return zip_code_index

    return None


def foo2():
    accident_prob = env.prob_acc[17]
    random_values = np.random.default_rng().random(len(accident_prob))
    for zip_code_index in range(len(accident_prob)):
        if random_values[zip_code_index] <= accident_prob[zip_code_index]:  # :
            return zip_code_index

    return None


# for i in range(10000):
# foo1()
# foo2()
