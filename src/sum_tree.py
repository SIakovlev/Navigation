import numpy as np


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.__capacity = capacity
        self.__tree = np.zeros(2 * capacity - 1)
        self.__data = np.zeros(capacity, dtype=object)
        self.__len = 0

    def __len__(self):
        return self.__len

    def __propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.__tree[parent] += change
        if parent:
            self.__propagate(parent, change)

    def __retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.__tree):
            return idx
        if s <= self.__tree[left]:
            return self.__retrieve(left, s)
        else:
            return self.__retrieve(right, s - self.__tree[left])

    def total(self):
        return self.__tree[0]

    def add(self, p, data):
        idx = self.write + self.__capacity - 1
        self.__data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.__capacity:
            self.write = 0

        if self.__len < self.__capacity:
            self.__len += 1

        return idx

    def update(self, idx, p):
        change = p - self.__tree[idx]
        self.__tree[idx] = p
        self.__propagate(idx, change)

    def get(self, s):
        idx = self.__retrieve(0, s)
        data_idx = idx - self.__capacity + 1
        return idx, self.__tree[idx], self.__data[data_idx]