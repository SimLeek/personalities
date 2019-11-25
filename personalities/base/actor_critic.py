import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from typing import List
import numpy as np
import math as m


class ActorCritic(nn.Module):
    # from: https://github.com/colinskow/move37/
    # todo: needs to be variational. Replace the nn.Linear with a new model based on
    #  this link: https://luiarthur.github.io/statorial/varinf/linregpy/
    #  Then, prune away neurons that aren't used, and reduce/expand
    #  when needed.
    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, max_complexity),
            nn.ReLU(),
            nn.Linear(max_complexity, 1),
        )

        self.critic._modules["0"].bias = torch.nn.Parameter(
            self.critic._modules["0"].bias / 2000
        )
        self.critic._modules["0"].weight = torch.nn.Parameter(
            self.critic._modules["0"].weight / 2000
        )
        self.critic._modules["2"].bias = torch.nn.Parameter(
            self.critic._modules["2"].bias / 2000
        )
        self.critic._modules["2"].weight = torch.nn.Parameter(
            self.critic._modules["2"].weight / 2000
        )

        self.num_outputs = num_outputs
        self.out_feedback = torch.zeros((1, num_outputs))

        self.actor = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, max_complexity),
            nn.ReLU(),
            nn.Linear(max_complexity, num_outputs),
        )

        self.actor._modules["0"].bias = torch.nn.Parameter(
            self.actor._modules["0"].bias / 2000
        )
        self.actor._modules["0"].weight = torch.nn.Parameter(
            self.actor._modules["0"].weight / 2000
        )
        self.actor._modules["2"].bias = torch.nn.Parameter(
            self.actor._modules["2"].bias / 2000
        )
        self.actor._modules["2"].weight = torch.nn.Parameter(
            self.actor._modules["2"].weight / 2000
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        if x.ndim == 2:
            if self.out_feedback.ndim == 3:
                self.out_feedback = self.out_feedback.to(x.device).squeeze(dim=0)
                if self.out_feedback.shape[0] == 1 and x.shape[0] > 1:
                    self.out_feedback = torch.cat(
                        [self.out_feedback] * x.shape[0], dim=0
                    )
                x = torch.cat((x, self.out_feedback), dim=1)
            else:
                x = torch.cat(
                    (
                        x,
                        torch.stack(
                            [self.out_feedback.to(x.device).squeeze()] * x.shape[0]
                        ),
                    ),
                    dim=1,
                )
        else:
            self.out_feedback = self.out_feedback.to(x.device).squeeze()
            if self.out_feedback.ndim > 1:
                self.out_feedback = self.out_feedback[-1, ...]  # get last of batch
            x = torch.cat((x, self.out_feedback))
        value = self.critic(x)
        mu = self.actor(x)
        if x.ndim == 2:
            expand = mu
        else:
            expand = mu.view(1, self.num_outputs)
        std = self.log_std.exp().expand_as(expand)
        dist = Normal(mu, std)
        self.out_feedback = dist.mean[None, :]
        self.out_feedback = self.out_feedback.detach()
        return dist, value


def standardize(x: torch.Tensor, machine_epsilon=1e-8):
    """
    Standardize a tensor so that all values in it have an overall mean of 0 and std of 1.

    This is sometimes called normalizing.

    :param x: The tensor we're standardizing.
    :param machine_epsilon: This will prevent us from dividing by zero if the std is small or zero.
    """
    x -= x.mean()
    x /= x.std() + machine_epsilon
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

    def update(self, reward, done):
        log_prob = self.last_dist.log_prob(self.last_action)
        self.log_probs.append(log_prob)
        self.values.append(self.last_value)
        self.rewards.append(
            reward.clone()
            .detach()
            .to(torch.float32)
            .unsqueeze(0)
            .unsqueeze(1)
            .to(self.device)
        )
        self.done_masks.append(
            (1 - torch.tensor(done)).to(torch.float32).unsqueeze(1).to(self.device)
        )
        self.states.append(self.last_state)
        self.actions.append(self.last_action)
        return self

    def compute_torch(self):
        if isinstance(self.gaes, list):
            self.gaes = torch.cat(self.gaes).detach().to(torch.float32).to(self.device)
        if isinstance(self.log_probs, list):
            self.log_probs = (
                torch.cat(self.log_probs).detach().to(torch.float32).to(self.device)
            )
        if isinstance(self.values, list):
            self.values = (
                torch.cat(self.values).detach().to(torch.float32).to(self.device)
            )
        if isinstance(self.states, list):
            self.states = torch.cat(self.states).to(torch.float32).to(self.device)
        if isinstance(self.actions, list):
            self.actions = torch.cat(self.actions).to(torch.float32).to(self.device)
        if isinstance(self.advantages, list):
            self.advantages = self.gaes.squeeze() - self.values.squeeze()
            self.advantages = (
                standardize(self.advantages).to(torch.float32).to(self.device)
            )

    def batch_iter(self, mini_batch_size=64):
        if isinstance(self.states, list):
            raise RuntimeError(
                "Please run compute_torch on array memory before using batch_iter."
            )
        batch_size = self.states.size(0)

        for _ in range(int(m.ceil(batch_size / mini_batch_size))):
            random_selection = np.random.randint(0, batch_size - 1, mini_batch_size)
            batch = _ArrayActorCriticMemoryBatch()
            batch.state = self.states[random_selection, :]
            batch.action = self.actions[random_selection, :]
            batch.old_log_prob = self.log_probs[random_selection, :]
            batch.gae = self.gaes[random_selection, :]
            batch.advantage = self.advantages[random_selection]
            yield batch


class _ArrayActorCriticMemoryBatch(object):
    state: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    gae: torch.Tensor
    advantage: torch.Tensor

    def __init__(self):
        pass


class ProximalActorCritic(object):
    """
    An actor critic that can memorize its recent decisions so it can learn
    to optimize long term tasks.

    >>> acwm = ProximalActorCritic(5, 2, 256)
    >>> acwm.memory.reset()
    >>> while True:
    >>>   for action in range(len(512)):
    ...     # ... compute state here ...
    ...     action = acwm.get_action(state)
    ...     mouse.x = action[0]
    ...     mouse.y = action[1]
    ...     # Compute reward here. For curiosity, use loss of an autoencoder.
    ...     acwm.memory.update(reward=[1,0], done=[0])
    >>>   acwm.update_ppo()
    """

    _LEARNING_RATE = 1e-4

    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0):
        self.actor_critic = ActorCritic(
            num_inputs, num_outputs, max_complexity, std=std
        )
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self._LEARNING_RATE
        )
        self.device = next(self.actor_critic.parameters()).device
        self.memory = _ArrayActorCriticMemory(self.device)

        self.debug = False

    def cuda(self):
        self.actor_critic.cuda()
        self.device = next(self.actor_critic.parameters()).device

    def cpu(self):
        self.actor_critic.cpu()
        self.device = next(self.actor_critic.parameters()).device

    def get_action(self, state):
        state: torch.Tensor = state.to(torch.float32).to(self.device)
        dist, value = self.actor_critic(state.squeeze())
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
                self.memory.rewards[step].to(self.device)
                + anticipation
                * self.memory.values[min(step + 1, len(self.memory.rewards) - 1)].to(
                    self.device
                )
                * self.memory.done_masks[step].to(self.device)
                - self.memory.values[step].to(self.device)
            )
            gae = (
                delta
                + anticipation
                * smoothing
                * self.memory.done_masks[step].to(self.device)
                * gae
            )
            # prepend to get correct order back
            self.memory.gaes.insert(0, gae + self.memory.values[step].to(self.device))

    def update_ppo(
        self, epochs=10, ppo_clip=0.2, critic_discount=0.5, entropy_beta=0.001
    ):
        self.compute_gae()
        self.memory.compute_torch()

        for _ in range(epochs):
            for mem in self.memory.batch_iter():
                dist, value = self.actor_critic(mem.state.squeeze().to(self.device))
                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(mem.action.to(self.device))

                ratio = (new_log_prob - mem.old_log_prob.to(self.device)).exp()
                actor_loss_1 = ratio * torch.stack(
                    [mem.advantage] * ratio.shape[-1], dim=-1
                ).to(self.device)
                actor_loss_2 = torch.clamp(
                    ratio, 1.0 - ppo_clip, 1.0 + ppo_clip
                ) * torch.stack([mem.advantage] * ratio.shape[-1], dim=-1).to(
                    self.device
                )
                actor_loss = -torch.min(actor_loss_1, actor_loss_2).mean()
                critic_loss = (mem.gae.to(self.device) - value).pow(2).mean()

                loss = (
                    critic_discount * critic_loss + actor_loss - entropy_beta * entropy
                )
                if self.debug:
                    print(loss.item())
                loss.detach_()
                loss.requires_grad = True

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, filename):
        torch.save(self.actor_critic.state_dict(), filename)
