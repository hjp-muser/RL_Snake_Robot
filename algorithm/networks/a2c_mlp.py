from torch import nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(in_dim, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class Value(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(Value, self).__init__()
        self.l1 = nn.Linear(in_dim, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        return x