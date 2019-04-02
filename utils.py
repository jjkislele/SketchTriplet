import time
from functools import wraps
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms

def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)