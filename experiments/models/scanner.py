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
        seq = torch.arange(self.filter_size**2).reshape([1, self.filter_size*self.filter_size])
        seq = self.scan(seq).reshape([self.filter_size, self.filter_size])
        print("Scan order:")
        print(seq)

    @abstractmethod
    def scan(self, patch2d):
        pass


class RasterScanner2D(Scanner2D):
    def scan(self, flattened):
        return flattened

class RandomScanner2D(Scanner2D):
    def __init__(self, filter_size):
        super().__init__(filter_size=filter_size)
        self.order = torch.randperm(filter_size**2)

    def scan(self, flattened):
        return flattened[:,self.order]

class PartialTransposeScanner2D(Scanner2D):
    def __init__(self, filter_size):
        super().__init__(filter_size=filter_size)
        condition = torch.tensor(np.indices([filter_size, filter_size]).sum(axis=0) % 2 == 1)
        self.order = torch.arange(filter_size**2).reshape([filter_size, filter_size])
        self.order = torch.where(condition, self.order.transpose(0,1), self.order).flatten()

    def scan(self, flattened):
        return flattened[:,self.order]


class ZigZagScanner2D(Scanner2D):
    def __init__(self, filter_size):
        super().__init__(filter_size=filter_size)
        self.order = [0, 1, 3, 6, 2, 4, 7, 10, 5, 8, 11, 13, 9, 12, 14, 15]

    def scan(self, flattened):
        return flattened[:,self.order]

print("RasterScanner2D")
RasterScanner2D(4).print()
print("RandomScanner2D")
RandomScanner2D(4).print()
print("PartialTransposeScanner2D")
PartialTransposeScanner2D(4).print()
print("ZigZagScanner2D")
ZigZagScanner2D(4).print()