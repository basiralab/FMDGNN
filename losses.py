import torch
import torch.nn.functional as F
import networkx as nx
import time

from utils import *
from config import *


def reconstruction_loss(pred, target):
    """Compute the reconstruction loss"""
    return F.l1_loss(pred, target)



