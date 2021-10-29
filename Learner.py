import torch
import numpy as np
from torch.nn.functional import pad

from Memory import Transition
from torch import tensor, device, cuda, bool, cat, zeros, stack, argmax, transpose, narrow
import time

BATCH_SIZE = 32
DEFAULT_DISCOUNT = 0.99

device = device("cuda" if cuda.is_available() else "cpu")


class Learner(object):
    def __init__(self, model):
        self.model = model
        self.loss_array = []

    def optimize_model(self, memory):
        """
        Trains the model
        :param memory:
        :return:
        """
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        start_time = time.time()
        next_state_batch_torch = cat([s.get_torch() for s in batch.next_state if s is not None])
        state_batch_torch = cat([s.get_torch() for s in batch.state if s is not None])
        action_batch = cat(batch.action)
        reward_batch = np.squeeze(cat(batch.reward)).clone().detach()

        step = batch.state[0].N
        not_mask_lengths = [len(s.indexNotMasked) for s in batch.state if s is not None]
        not_mask = cat([tensor(s.indexNotMasked+i*step, device=device) for i, s in enumerate(batch.state) if s is not None])

        # This happens to take the correct action in gather
        for i, action in enumerate(action_batch):
            action[0] += i*step
        q_values = self.model.policy_net(state_batch_torch).gather(0, action_batch).squeeze(dim=1)
        next_q_values_all = self.model.target_net(next_state_batch_torch).gather(0, not_mask.unsqueeze(1))

        # Here we find max element of each slice
        index = 0
        next_q_values_max = tensor([], device=device)

        for length in not_mask_lengths:
            slice = narrow(next_q_values_all, 0, index, length)
            index += length
            if length == 0:
                max_q_value = 0
            else:
                max_q_value = slice.max()
            next_q_values_max = torch.cat((next_q_values_max, tensor([max_q_value], device=device)), 0)

        # Compute the expected Q values
        expected_q_values = (next_q_values_max * DEFAULT_DISCOUNT) + reward_batch
        optimizer = self.model.optimizer
        loss = self.model.loss_fn(expected_q_values, q_values)

        self.loss_array.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        clip_value = 0.1
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), clip_value)
        optimizer.step()
