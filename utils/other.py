import random
import numpy
try:
    import torch
except ImportError:
    print("no torch")

import collections


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except ImportError:
        print("no torch")