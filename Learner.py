import torch
import numpy as np
from torch.nn.functional import pad

from Memory import Transition
from torch import tensor, device, cuda, bool, cat, zeros, stack, argmax, transpose, narrow, int64
import time

BATCH_SIZE = 16
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

        self.model.optimizer.zero_grad()

        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        start_time = time.time()
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        next_state_batch_torch = cat([s.get_torch() for s in batch.next_state if s is not None])
        state_batch_torch = cat([s.get_torch() for s in batch.state if s is not None])


        action_batch = cat(batch.action)
        reward_batch = np.squeeze(cat(batch.reward)).clone().detach()

        step = batch.state[0].N

        not_mask_lengths = [len(s.indexNotMasked) for s in batch.state if s is not None]
        not_mask = tensor([], device=device)
        count = 0
        for i, s in enumerate(batch.state):
            if s is not None and non_final_mask[i]:
                not_mask = cat((not_mask, tensor(s.indexNotMasked+count*step, device=device, dtype=int64)), 0)
                count += 1

        # This happens to take the correct action in gather
        for i, action in enumerate(action_batch):
            action[0] += i*step
        q_values = self.model.policy_net(state_batch_torch)
        q_values = q_values.squeeze(1).gather(0, action_batch).squeeze(dim=1)
        not_mask = not_mask.type(int64)
        next_q_values_all = self.model.target_net(next_state_batch_torch)
        next_q_values_all = next_q_values_all.squeeze(1).gather(0, not_mask.unsqueeze(1))

        # Here we find max element of each slice
        index = 0
        next_q_values_max = tensor([], device=device)

        for i,length in enumerate(not_mask_lengths):
            if not non_final_mask[i] or length == 0:
                max_q_value = 0
            else:
                slice = narrow(next_q_values_all, 0, index, length)
                #if len(slice[slice != slice[0]]) > 0:
                    #print("learner")
                index += length
                max_q_value = slice.max()
            next_q_values_max = torch.cat((next_q_values_max, tensor([max_q_value], device=device)), 0)

        # Compute the expected Q values
        expected_q_values = (next_q_values_max * DEFAULT_DISCOUNT) + reward_batch
        loss = self.model.loss_fn(expected_q_values, q_values)

        self.loss_array.append(loss.item())

        self.model.optimizer.zero_grad()
        loss.backward()
        clip_value = 0.1
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), clip_value)
        self.model.optimizer.step()
