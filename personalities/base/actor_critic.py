import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from typing import List
import numpy as np


class ActorCritic(nn.Module):
    # from: https://github.com/colinskow/move37/
    # todo: needs to be variational. Replace the nn.Linear with a new model based on
    #  this link: https://luiarthur.github.io/statorial/varinf/linregpy/
    #  Then, prune away neurons that aren't used, and reduce/expand
    #  when needed.
    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, max_complexity),
            nn.ReLU(),
            nn.Linear(max_complexity, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, max_complexity),
            nn.ReLU(),
            nn.Linear(max_complexity, num_outputs),
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


def standardize(x: torch.Tensor, machine_epsilon=1e-8):
    """
    Standardize a tensor so that all values in it have an overall mean of 0 and std of 1.

    This is sometimes called normalizing.

    :param x: The tensor we're standardizing.
    :param machine_epsilon: This will prevent us from dividing by zero if the std is small or zero.
    """
    x -= x.mean()
    x /= (x.std() + machine_epsilon)
    return x


class _ArrayActorCriticMemory(object):
    def __init__(self, device):
        self.device = device

        self.last_action = None
        self.last_dist = None
        self.last_value = None
        self.last_state = None

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.done_masks = []
        self.states = []
        self.actions = []
        self.gaes: List[torch.Tensor] = []
        self.advantages: List[torch.Tensor] = []

    def reset(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.done_masks = []
        self.states = []
        self.actions = []
        return self

    def update(self, device, reward, done):
        log_prob = self.last_dist.log_prob(self.last_action)
        self.log_probs.append(log_prob)
        self.values.append(self.last_value)
        self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        self.done_masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        self.states.append(self.last_state)
        self.actions.append(self.last_action)
        return self

    def compute_torch(self):
        if isinstance(self.gaes, list):
            self.gaes = torch.cat(self.gaes).detach()
        if isinstance(self.log_probs, list):
            self.log_probs = torch.cat(self.log_probs).detach()
        if isinstance(self.values, list):
            self.values = torch.cat(self.values).detach()
        if isinstance(self.states, list):
            self.states = torch.cat(self.states)
        if isinstance(self.actions, list):
            self.actions = torch.cat(self.actions)
        if isinstance(self.advantages, list):
            self.advantages = self.gaes - self.values
            self.advantages = standardize(self.advantages)

    def batch_iter(self, mini_batch_size=64):
        if isinstance(self.states, list):
            raise RuntimeError("Please run compute_torch on array memory before using batch_iter.")
        batch_size = self.states.size(0)

        for _ in range(batch_size // mini_batch_size):
            random_selection = np.random.randint(0, batch_size, mini_batch_size)
            batch = _ArrayActorCriticMemoryBatch()
            batch.state = self.states[random_selection, :]
            batch.action = self.actions[random_selection, :]
            batch.old_log_prob = self.log_probs[random_selection, :]
            batch.gae = self.gaes[random_selection, :]
            batch.advantage = self.advantages[random_selection, :]
            yield batch


class _ArrayActorCriticMemoryBatch(object):
    state: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    gae: torch.Tensor
    advantage: torch.Tensor

    def __init__(self):


class ProximalActorCritic(object):
    """
    An actor critic that can memorize its recent decisions so it can learn
    to optimize long term tasks.

    >>> acwm = ProximalActorCritic(5, 2, 256)
    >>> acwm.memory.reset()
    >>> for action in range(len(512)):
    ...   # ... compute state here ...
    ...   action = acwm.get_action(state)
    ...   mouse.x = action[0]
    ...   mouse.y = action[1]
    ...   # ... compute reward here ...
    ...   acwm.memory.update(reward=[1,0], done=[0])
    >>> acwm.update_ppo()
    """
    _LEARNING_RATE = 1e-4

    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0):
        self.actor_critic = ActorCritic(num_inputs, num_outputs, max_complexity, std=std)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self._LEARNING_RATE)
        self.device = next(self.actor_critic.parameters()).device
        self.memory = _ArrayActorCriticMemory(self.device)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist, value = self(state)
        self.memory.last_dist = dist
        self.memory.last_value = value
        action = dist.sample()
        self.memory.last_action = action
        self.memory.last_state = state
        return action

    def compute_gae(self, anticipation=0.99, smoothing=0.95):
        """
        Compute the generalized advantage estimate.

        This smooths out the learning over all the decisions we remember making,
        so we can learn to make long term decisions.

        :param anticipation:
        :param smoothing:
        :return:
        """
        gae = 0
        self.memory.gaes = []
        for step in reversed(range(len(self.memory.rewards))):
            delta = (
                    self.memory.rewards[step]
                    + anticipation
                    * self.memory.values[step + 1]
                    * self.memory.done_masks[step]
                    - self.memory.values[step]
            )
            gae = delta + anticipation * smoothing * self.memory.done_masks[step] * gae
            # prepend to get correct order back
            self.memory.gaes.insert(0, gae + self.memory.values[step])

    def update_ppo(self, epochs=10, ppo_clip=0.2, critic_discount=0.5, entropy_beta=0.001):
        self.compute_gae()
        self.memory.compute_torch()

        for _ in range(epochs):
            for mem in self.memory.batch_iter():
                dist, value = self.actor_critic(mem.state)
                entropy = dist.entropy.mean()
                new_log_prob = dist.log_prob(mem.action)

                ratio = (new_log_prob - mem.old_log_prob).exp()
                actor_loss_1 = ratio * mem.advantage
                actor_loss_2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * mem.advantage
                actor_loss = - torch.min(actor_loss_1, actor_loss_2).mean()
                critic_loss = (mem.gae - value).pow(2).mean()

                loss = critic_discount * critic_loss + actor_loss - entropy_beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, filename):
        torch.save(self.actor_critic.state_dict(), filename)
