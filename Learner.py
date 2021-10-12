from Memory import Transition
from Memory import ReplayMemory
from torch import tensor, device, cuda, bool, cat, zeros
BATCH_SIZE = 128
DEFAULT_DISCOUNT = 0.99

device = device("cuda" if cuda.is_available() else "cpu")


class Learner(object):
    def __init__(self, model):

        self.model = model

    def optimize_model(self, memory):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=bool)
        non_final_next_states = cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = cat(batch.state)
        action_batch = cat(batch.action)
        reward_batch = cat(batch.reward)

        q_values = self.model.policy_net(state_batch).gather(1, action_batch)

        next_q_values = zeros(BATCH_SIZE, device=device)
        next_q_values[non_final_mask] = self.model.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_q_values = (next_q_values * DEFAULT_DISCOUNT) + reward_batch

        optimizer = self.model.policy_net.optimizer
        loss = self.model.policy_net.loss_fn(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.model.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
