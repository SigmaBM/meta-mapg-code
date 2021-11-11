from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.torch_utils import weight_init


class ActorNetwork(nn.Module):
    """Actor network with MLP that outputs action
    Args:
        input_dim (int): Input dimension to network
        output_dim (int): Output dimension of network
        name (str): Prefix for each layer
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, input_dim, output_dim, name, args):
        super(ActorNetwork, self).__init__()

        setattr(self, name + "_actor_l1", nn.Linear(input_dim, args.n_hidden))
        setattr(self, name + "_actor_l2", nn.Linear(args.n_hidden, args.n_hidden))
        setattr(self, name + "_actor_l3", nn.Linear(args.n_hidden, output_dim))
        self.name = name + "_actor"
        self.apply(weight_init)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = F.relu(F.linear(x, weight=params[self.name + "_l1.weight"], bias=params[self.name + "_l1.bias"]))
        x = F.relu(F.linear(x, weight=params[self.name + "_l2.weight"], bias=params[self.name + "_l2.bias"]))
        x = F.linear(x, weight=params[self.name + "_l3.weight"], bias=params[self.name + "_l3.bias"])
        x = F.softmax(x, dim=1)

        return x


class ValueNetwork(nn.Module):
    """Value network with MLP that outputs value (V)
    Args:
        input_dim (int): Input dimension to network
        name (str): Prefix for each layer
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, input_dim, name, args):
        super(ValueNetwork, self).__init__()

        setattr(self, name + "_value_l1", nn.Linear(input_dim, args.n_hidden))
        setattr(self, name + "_value_l2", nn.Linear(args.n_hidden, args.n_hidden))
        setattr(self, name + "_value_l3", nn.Linear(args.n_hidden, 1))
        self.name = name + "_value"
        self.apply(weight_init)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = F.relu(F.linear(x, weight=params[self.name + "_l1.weight"], bias=params[self.name + "_l1.bias"]))
        x = F.relu(F.linear(x, weight=params[self.name + "_l2.weight"], bias=params[self.name + "_l2.bias"]))
        x = F.linear(x, weight=params[self.name + "_l3.weight"], bias=params[self.name + "_l3.bias"])

        return x
