import random
import torch
import numpy as np
from sum_tree import SumTree

from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    """
    Short term memory buffer
    Data is sampled all at once and gets removed from memory
    """

    def __init__(self, buffer_size, batch_size, seed):
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        #random.seed(seed)
        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__memory_buffer = []
        self.__memory = SumTree(buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.__memory_buffer.append(self.__experience(state, action, reward, next_state, done))

    def sample(self):

        buf_len = len(self.__memory_buffer)
        mem_len = self.__batch_size - buf_len

        experiences = []
        indices = []
        probs = []

        if mem_len:
            segment = self.__memory.total() / mem_len
            for i in range(mem_len):
                #s = random.uniform(segment * i, segment * (i + 1))
                s = random.uniform(0, self.__memory.total())
                idx, p, e = self.__memory.get(s)
                experiences.append(e)
                indices.append(idx)
                probs.append(p/self.__memory.total())

        for e in self.__memory_buffer:
            # Add experience to the buffer and record its index
            idx = self.__memory.add(0.0, e) # Default value for p is 0
            experiences.append(e)
            indices.append(idx)
            probs.append(1/len(self))

        self.__memory_buffer.clear()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones, indices, probs

    def update(self, indices, p_values):
        for idx, p in zip(indices, p_values):
            self.__memory.update(idx, p)

    def __len__(self):
        return max(len(self.__memory), len(self.__memory_buffer))
