import logging

from abc import ABC, abstractmethod

import torch
import numpy as np

# Base class
class Scanner2D(ABC):
    def __init__(self, filter_size):
        self.filter_size = filter_size
        super().__init__()

    def print(self):
        seq = torch.arange(self.filter_size**2).reshape([self.filter_size, self.filter_size, 1])
        seq = self.scan(seq).reshape([self.filter_size, self.filter_size])
        print("Scan order:")
        print(seq)

    @abstractmethod
    def scan(self, patch2d):
        pass


class RasterScanner2D(Scanner2D):
    def scan(self, patch2d):
        flattened = patch2d.flatten()
        return flattened

class RandomScanner2D(Scanner2D):
    def __init__(self, filter_size):
        super().__init__(filter_size=filter_size)
        self.rand_idx = torch.randperm(filter_size**2)

    def scan(self, patch2d):
        flattened = patch2d.flatten()[self.rand_idx]
        return flattened

class PartialTransposeScanner2D(Scanner2D):
    def scan(self, patch2d):
        condition = torch.tensor(np.indices([self.filter_size, self.filter_size, 1]).sum(axis=0) % 2 == 1)
        condition = condition.expand(patch2d.size())
        patch2d = torch.where(condition, patch2d.transpose(0,1), patch2d)
        return patch2d.flatten()


class ZigZagScanner2D(Scanner2D):
    def __init__(self, filter_size):
        super().__init__(filter_size=filter_size)
        self.order = [0, 1, 3, 6, 2, 4, 7, 10, 5, 8, 11, 13, 9, 12, 14, 15]

    def scan(self, patch2d):
        flattened = patch2d.flatten()[self.order]
        return flattened

# ZigZagScanner2D(4).print()