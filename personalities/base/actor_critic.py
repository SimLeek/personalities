import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from typing import List
import numpy as np
import math as m
from .odst import ODST
from .nn_utils import Lambda


class ActorCritic(nn.Module):
    """
    An actor critic model. The actor neural network decides which outputs to give given the current inputs
    (environment, memory, etc.) The critic rates how good that action was, and can be used for training the actor.
    """

    # from: https://github.com/colinskow/move37/
    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0):
        """
        An actor critic model. The actor neural network decides which outputs to give given the current inputs
        (environment, memory, etc.) The critic rates how good that action was, and can be used for training the actor.

        :param num_inputs: number of float input numbers used to represent the current state or other input factors.
        :param num_outputs: max number of outputs the actor can give
        :param max_complexity: number of contrasting options there are to reach the goal. Also number of trees in
        networks.
        :param std: standard deviation of the output
        """
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            ODST(
                in_features=num_inputs + num_outputs + 1,
                num_trees=max_complexity,
                tree_dim=1,
                flatten_output=False,
                depth=6,
            ),
            Lambda(lambda x: x.mean(1)),
        )

        self.num_outputs = num_outputs
        self.out_feedback = torch.zeros((1, num_outputs))
        self.val_feedback = torch.zeros((1, 1))

        self.actor = nn.Sequential(
            ODST(
                in_features=num_inputs + num_outputs + 1,
                num_trees=max_complexity,
                tree_dim=num_outputs,
                flatten_output=False,
                depth=6,
            ),
            Lambda(lambda x: x.mean(1)),
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def _append_prev_val_to_x(self, x: torch.Tensor):
        """
        Append the previous output tensor to the input state.

        Note: this must be accounted for in init by changing the input size.
        """
        if x.ndimension() == 2:
            if self.val_feedback.ndim == 3:
                self.val_feedback = self.val_feedback.to(x.device)
                if self.val_feedback.shape[0] == 1 and x.shape[0] > 1:
                    self.val_feedback = torch.cat(
                        [self.val_feedback] * x.shape[0], dim=0
                    )
                x = torch.cat((x, self.val_feedback), dim=1)
            else:
                val_fb = self.val_feedback.type_as(x).squeeze()
                if val_fb.ndim == 0:
                    val_fb = val_fb.unsqueeze(0)
                x = torch.cat((x, torch.stack([val_fb] * x.shape[0], dim=0)), dim=1)
        else:
            if self.val_feedback.ndim > 1:
                self.val_feedback = self.val_feedback[-1, ...]
            x = torch.cat((x, self.val_feedback.type_as(x).squeeze().unsqueeze(0)))
        return x

    def _append_prev_out_to_x(self, x: torch.Tensor):
        """
        Append the previous output tensor to the input state.

        Note: this must be accounted for in init by changing the input size.
        """

        if x.ndimension() == 2:
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
        return x

    def forward(self, x):
        x = self._append_prev_out_to_x(x)
        x = self._append_prev_val_to_x(x)
        with torch.no_grad():
            value = self.critic(x)
            mu = self.actor(x)
        if x.ndimension() == 2:
            expand = mu
        else:
            expand = mu.view(1, self.num_outputs)
        std = self.log_std.exp().expand_as(expand)
        dist = Normal(mu, std)
        self.out_feedback = dist.mean[None, :]
        self.out_feedback = self.out_feedback.detach()
        self.val_feedback = value.detach()
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


class _QueueActorCriticMemory(object):
    """Holds memory for an actor critic learner."""

    def __init__(self, max_size=float("inf")):
        """
        Holds memory for an actor critic module.

        :param device: cpu or gpu device to run on.
        :param max_size: Max size of the memory. Note: this needs to be finite for PPO to learn.
        """
        self.device = None

        self.last_action = None
        self.last_dist = None
        self.last_value = None
        self.last_state = None

        self.log_probs = torch.empty((0,))
        self.values = torch.empty((0,))
        self.rewards = torch.empty((0,))
        self.done_masks = torch.empty((0,))
        self.states = torch.empty((0,))
        self.actions = torch.empty((0,))
        self.gaes: torch.Tensor = torch.empty((0,))
        self.advantages: torch.Tensor = torch.empty((0,))

        self.max_size = max_size

    def to(self, device):
        self.device = device
        return self

    def serialize(self):
        ser = {
            "last_action": self.last_action,
            "last_dist": self.last_dist,
            "last_value": self.last_value,
            "last_state": self.last_state,
            "log_probs": self.log_probs,
            "values": self.values,
            "rewards": self.rewards,
            "done_masks": self.done_masks,
            "states": self.states,
            "actions": self.actions,
            "gaes": self.gaes,
            "advantages": self.advantages,
            "max_size": self.max_size,
        }
        return ser

    @classmethod
    def deserialize(cls, cereal):
        de = cls(cereal["max_size"])

        de.last_action = cereal["last_action"]
        de.last_dist = cereal["last_dist"]
        de.last_value = cereal["last_value"]
        de.last_state = cereal["last_state"]
        de.log_probs = cereal["log_probs"]
        de.values = cereal["values"]
        de.rewards = cereal["rewards"]
        de.done_masks = cereal["done_masks"]
        de.states = cereal["states"]
        de.actions = cereal["actions"]
        de.gaes = cereal["gaes"]
        de.advantages = cereal["advantages"]

        return de

    def __len__(self):
        return self.values.numel()

    def cuda(self):
        """Change all memory to run on a cuda device."""
        self.log_probs.cuda()
        self.values.cuda()
        self.rewards.cuda()
        self.done_masks.cuda()
        self.states.cuda()
        self.actions.cuda()
        self.gaes.cuda()
        self.advantages.cuda()
        self.device = self.values.device

    def cpu(self):
        """Change all memory to run on a cpu."""
        self.log_probs.cpu()
        self.values.cpu()
        self.rewards.cpu()
        self.done_masks.cpu()
        self.states.cpu()
        self.actions.cpu()
        self.gaes.cpu()
        self.advantages.cpu()
        self.device = self.values.device

    def reset(self):
        """Reset all memory to empty queues."""
        self.log_probs = torch.empty((0,)).to(self.device)
        self.values = torch.empty((0,)).to(self.device)
        self.rewards = torch.empty((0,)).to(self.device)
        self.done_masks = torch.empty((0,)).to(self.device)
        self.states = torch.empty((0,)).to(self.device)
        self.actions = torch.empty((0,)).to(self.device)
        return self

    def update(self, reward, done):
        """Add a memory to the queue. Pop front if there are more than max_size memories."""
        log_prob = self.last_dist.log_prob(self.last_action).to(self.device)
        if len(self) < self.max_size:
            self.log_probs = torch.cat([self.log_probs, log_prob])
            self.values = torch.cat([self.values, self.last_value.to(self.device)])
            self.rewards = torch.cat(
                [
                    self.rewards,
                    torch.tensor([reward])
                    .detach()
                    .to(torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .to(self.device),
                ]
            )
            self.done_masks = torch.cat(
                [
                    self.done_masks,
                    (1 - torch.tensor(done))
                    .to(torch.float32)
                    .unsqueeze(1)
                    .to(self.device),
                ]
            )
            self.states = torch.cat([self.states, self.last_state.to(self.device)])
            self.actions = torch.cat([self.actions, self.last_action.to(self.device)])
        else:
            self.log_probs = torch.cat([self.log_probs[1:], log_prob])
            self.values = torch.cat([self.values[1:], self.last_value.to(self.device)])
            self.rewards = torch.cat(
                [
                    self.rewards[1:],
                    torch.tensor([reward])
                    .detach()
                    .to(torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .to(self.device),
                ]
            )
            self.done_masks = torch.cat(
                [
                    self.done_masks[1:],
                    (1 - torch.tensor(done))
                    .to(torch.float32)
                    .unsqueeze(1)
                    .to(self.device),
                ]
            )
            self.states = torch.cat([self.states[1:], self.last_state.to(self.device)])
            self.actions = torch.cat(
                [self.actions[1:], self.last_action.to(self.device)]
            )
        return self

    def batch_iter(self, mini_batch_size=64):
        """Iterate through random batches of memories"""
        batch_size = self.states.size(0)

        self.advantages = self.gaes.squeeze() - self.values.squeeze()
        self.advantages = standardize(self.advantages).to(torch.float32).to(self.device)

        for _ in range(int(m.ceil(batch_size / mini_batch_size))):
            random_selection = np.random.randint(0, batch_size - 1, mini_batch_size)
            batch = _ArrayActorCriticMemoryBatch()
            batch.state = self.states[random_selection, :]
            batch.action = self.actions[random_selection, :]
            batch.old_log_prob = self.log_probs[random_selection, :]
            batch.gae = self.gaes[random_selection]
            batch.advantage = self.advantages[random_selection]
            yield batch


class _ArrayActorCriticMemoryBatch(object):
    """A single batch of actor critic memory."""

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
    """

    _LEARNING_RATE = 1e-4

    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0):
        """Create an actor critic that can optimize for long term tasks."""
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._max_complexity = max_complexity
        self._std = std

        self.actor_critic: nn.Module = ActorCritic(
            num_inputs, num_outputs, max_complexity, std=std
        )
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self._LEARNING_RATE
        )
        self.device = next(self.actor_critic.parameters()).device
        self.memory = _QueueActorCriticMemory().to(self.device)

        self.debug = False

    @property
    def ready_to_update(self):
        return self.memory.last_dist is not None

    def serialize(self):
        state = {
            "num_inputs": self._num_inputs,
            "num_outputs": self._num_outputs,
            "max_complexity": self._max_complexity,
            "std": self._std,
            "actor_critic_state": self.actor_critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "memory": self.memory.serialize(),
        }
        return state

    @classmethod
    def deserialize(cls, cereal):
        de = cls(
            num_inputs=cereal["num_inputs"],
            num_outputs=cereal["num_outputs"],
            max_complexity=cereal["max_complexity"],
            std=cereal["std"],
        )
        de.actor_critic.load_state_dict(cereal["actor_critic_state"])
        de.optimizer.load_state_dict(cereal["optimizer_state"])
        de.memory = de.memory.deserialize(cereal["memory"])

        return de

    def cuda(self):
        """Set every part of this model to run on a cuda device."""
        self.actor_critic.cuda()
        self.memory.cuda()
        self.device = next(self.actor_critic.parameters()).device

    def cpu(self):
        """Set every part of this model to run on a cpu device."""
        self.actor_critic.cpu()
        self.memory.cpu()
        self.device = next(self.actor_critic.parameters()).device

    def get_action(self, state):
        """Get the action the actor will take for a given state."""
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
        if self.memory.gaes.numel() > 0:
            gae = self.memory.gaes[-1].item()
        else:
            gae = 0
        for step in reversed(
            range(self.memory.gaes.numel() - 1, self.memory.rewards.numel())
        ):
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
            self.memory.gaes = torch.cat(
                [
                    self.memory.gaes.to(self.memory.device),
                    gae.to(self.memory.device)
                    + self.memory.values[step].to(self.memory.device),
                ]
            )
            if self.memory.gaes.numel() > self.memory.rewards.numel():
                self.memory.gaes = self.memory.gaes[1:]

    def update_ppo(
        self, epochs=10, ppo_clip=0.2, critic_discount=0.5, entropy_beta=0.001
    ):
        self.compute_gae()

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
                loss.detach_()
                loss.requires_grad = True

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, filename):
        torch.save(self.serialize(), filename)


class ContinualProximalActorCritic(ProximalActorCritic):
    """An actor critic that can learn to optimize long term, sparse reward, without stopping every ~200 frames."""

    def __init__(self, num_inputs, num_outputs, max_complexity, std=0.0, memory_len=8):
        """Create an actor critic that can continually optimize for long term tasks."""

        super().__init__(num_inputs, num_outputs, max_complexity, std)
        self.memory_len = memory_len
        self.memory.max_size = memory_len

    def update_ppo(
        self, epochs=1, ppo_clip=0.2, critic_discount=0.5, entropy_beta=0.001
    ):
        if len(self.memory) >= self.memory_len:
            self.compute_gae()

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
                        critic_discount * critic_loss
                        + actor_loss
                        - entropy_beta * entropy
                    )
                    print(loss.item())
                    loss.detach_()
                    loss.requires_grad = True

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
