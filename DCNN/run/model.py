import os
import gc
import time
import random

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from torch.autograd import Variable

class NodeClassDCNN(nn.Module):
    def __init__(self, param):
        super(NodeClassDCNN, self).__init__()
        self.H = param['n_hop']
        self.F = param['feat_shape']
        self.N = param['num_nodes']
        self.n_class = param['num_class']
        
        self.Wc = nn.Parameter(torch.FloatTensor(self.H+1, self.F))
        
        self.dense = nn.Linear((self.H+1)*self.F, self.n_class)
    
    def forward(self, X, P):# shape of X:N*F; shape of P:bs*H*N
        Z = P@X # bs*H*F
        embed = torch.tanh(Z*self.Wc)# bs*(H+1)*F
        embed = embed.view(embed.size(0), -1)
        out = torch.tanh(self.dense(embed))
        
        return out