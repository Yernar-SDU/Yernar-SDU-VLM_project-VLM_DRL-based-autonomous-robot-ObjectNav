#!/usr/bin/env python3
import numpy as np
import random

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) for our single-channel image + 7 scalars.
    We'll store: (img, scalars, action, reward, done, next_img, next_scalars, priority).
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.epsilon = 1e-5  # to avoid 0 priority

    def add(self, state_img, state_scl, action, reward, done, next_img, next_scl):
        """
        state_img: (1,64,64) np.float32
        state_scl: (7,)    np.float32
        action: (action_dim,)
        reward: float
        done: float
        next_img, next_scl: same shapes as above
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        transition = (state_img, state_scl, action, reward, done, next_img, next_scl)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None

        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[: len(self.buffer)])

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch_img = []
        batch_scl = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_next_img = []
        batch_next_scl = []

        for (s_img, s_scl, a, r, d, ns_img, ns_scl) in samples:
            batch_img.append(s_img)
            batch_scl.append(s_scl)
            batch_actions.append(a)
            batch_rewards.append(r)
            batch_dones.append(d)
            batch_next_img.append(ns_img)
            batch_next_scl.append(ns_scl)

        return (
            np.array(batch_img, dtype=np.float32),      # (batch,1,64,64)
            np.array(batch_scl, dtype=np.float32),      # (batch,7)
            np.array(batch_actions, dtype=np.float32),  # (batch, action_dim)
            np.array(batch_rewards, dtype=np.float32),  # (batch,)
            np.array(batch_dones, dtype=np.float32),    # (batch,)
            np.array(batch_next_img, dtype=np.float32), # (batch,1,64,64)
            np.array(batch_next_scl, dtype=np.float32), # (batch,7)
            weights,
            indices
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + self.epsilon

    def __len__(self):
        return len(self.buffer)
