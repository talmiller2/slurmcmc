import os, json, time
import numpy as np
from slurmpy import Slurm


class FunctionWrapper():
    def __init__(self, function):
        self.num_calls = 0
        self.points_history = []
        self.values_history = []
        self.function = function

    def __call__(self, x):
        res = self.function(x)
        self.num_calls += 1
        self.points_history += [x]
        self.values_history += [res]
        return res
