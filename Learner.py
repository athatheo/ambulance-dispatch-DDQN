import torch
import numpy as np
from Memory import Transition
from torch import tensor, device, cuda, bool, cat, zeros, stack, argmax

BATCH_SIZE = 64
DEFAULT_DISCOUNT = 0.99

device = device("cuda" if cuda.is_available() else "cpu")


class Learner(object):
    def __init__(self, model):
        self.model = model
        self.loss = 9999999999
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

        next_states = [s for s in batch.next_state if s is not None]
        next_state_batch_torch = cat([s.get_torch() for s in batch.next_state if s is not None])

        state_batch = [s for s in batch.state if s is not None]
        state_batch_torch = cat([s.get_torch() for s in batch.state if s is not None])

        action_batch = cat(batch.action)

        reward_batch = np.squeeze(cat(batch.reward)).clone().detach()

        q_values = self.model.policy_net(state_batch_torch)
        q_values = q_values.gather(0, action_batch).squeeze(dim=1)
        #q_values = self.get_q_vals(state_batch, target=False)
        next_q_values = self.get_q_vals(next_states, target=True).detach()
        #next_q_values = next_q_values.unsqueeze(1)

        # Compute the expected Q values
        expected_q_values = (next_q_values * DEFAULT_DISCOUNT) + reward_batch
        optimizer = self.model.optimizer

        loss = self.model.loss_fn(q_values, expected_q_values)

        self.loss_array.append(loss.item())
        if self.loss > loss.item() and self.loss < 20000:
            print(loss.item())
            self.loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        clip_value = 0.1
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), clip_value)
        optimizer.step()

    def get_q_vals(self, state_batch, target):
        """
        Returns a list of max q values
        :param model: current model
        :param state_batch: a list of states
        :return:
        """
        q_vals = [0 for _ in range(BATCH_SIZE)]
        for i, state in enumerate(state_batch):
            if target:
                q_vals[i] = self.get_q_max_target(state)
            else:
                q_vals[i] = self.get_q_max_policy(state)


        return stack(q_vals)

    def get_q_max_target(self, state):
        """
        Given the model and a state it return the max q value
        :param model:
        :param state:
        :return:
        """
        qvals = self.model.target_net(state.get_torch())
        state.indexNotMasked = np.sort(state.indexNotMasked)
        qvals_selectable = [qvals[i] for i in range(len(qvals)) if i in state.indexNotMasked]
        if len(qvals_selectable) == 0:
            return tensor(-1, device=device)
        qvals_selectable = stack(qvals_selectable)
        return torch.max(qvals_selectable)

    def get_q_max_policy(self, state):
        """
        Given the model and a state it return the max q value
        :param model:
        :param state:
        :return:
        """
        qvals = self.model.policy_net(state.get_torch())
        state.indexNotMasked = np.sort(state.indexNotMasked)
        qvals_selectable = [qvals[i] for i in range(len(qvals)) if i in state.indexNotMasked]
        if len(qvals_selectable) == 0:
            return tensor(-1, device=device)
        qvals_selectable = stack(qvals_selectable)
        return torch.max(qvals_selectable)