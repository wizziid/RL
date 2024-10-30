from collections import OrderedDict

import torch
from torch.distributions import Categorical, Bernoulli, Normal

class Value(torch.nn.Module):

    """
    A basic value function class with generic structure for all other policies.
    """
    
    def __init__(self, n_hidden, n_nodes, input_dim, output_dim=1):
        super().__init__()

        layers = []

        layers.append(("layer0", torch.nn.Linear(input_dim, n_nodes)))
        layers.append(("activation0", torch.nn.ReLU()))

        for l in range(n_hidden):
            layers.append((f"layer{l+1}", torch.nn.Linear(n_nodes, n_nodes)))
            layers.append((f"activation{l+1}", torch.nn.ReLU()))   

        layers.append(("layerOut", torch.nn.Linear(n_nodes, 1)))
        layers.append(("activationOut", torch.nn.Identity()))   

        self.network = torch.nn.Sequential(OrderedDict(layers))  


    def forward(self, obs):
        return self.network(obs)
    


