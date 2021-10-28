from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

RMSIZE = 1000

class ReplayMemory(object):

    def __init__(self):
        self.memory = deque([],maxlen=RMSIZE)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        #random.sample(self.memory[:100], batch_size-28)
        #random.sample(self.memory[100:], batch_size-100)
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)
