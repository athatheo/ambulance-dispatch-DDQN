import Environment
import shelve
from State import State
import torch

env = Environment.Environment()
#env.import_data()
#print(env.nr_ambulances)
#max_key = max(env.nr_ambulances, key= lambda x: env.nr_ambulances[x])
#print(env.nr_ambulances[max_key])
environment_data = shelve.open('environment.txt')
env = environment_data['key']
environment_data.close()

print(env.postcode_dic[1])
print("00000000000000000000000000000")

from torch import cuda

print(cuda.is_available())
state = State(env, 1)
x = torch.tensor([state.bool_accident,
                             state.nr_ambulances,
                             state.is_base,
                             state.travel_time,
                             state.delta,
                             state.time])
print(x.shape)
print(state.K, state.N)
print("Ambulances left: ", sum(state.nr_ambulances))
