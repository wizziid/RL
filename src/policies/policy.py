from collections import OrderedDict

import torch
from torch.distributions import Categorical, Bernoulli, Normal

class Policy(torch.nn.Module):

    """
    A basic policy class with generic structure for all other policies.
    """
    
    def __init__(self, n_hidden, n_nodes, input_dim, output_dim, distribution):
        super().__init__()

        layers = []

        layers.append(("layer0", torch.nn.Linear(input_dim, n_nodes)))
        layers.append(("activation0", torch.nn.ReLU()))

        for l in range(n_hidden):
            layers.append((f"layer{l+1}", torch.nn.Linear(n_nodes, n_nodes)))
            layers.append((f"activation{l+1}", torch.nn.ReLU()))   

        if distribution=="categorical":
            output_activation = torch.nn.Softmax(dim=-1)
        elif distribution =="normal":
            output_activation = torch.nn.Identity()
        else:
            output_activation = torch.nn.Sigmoid()

        layers.append(("layerOut", torch.nn.Linear(n_nodes, output_dim)))
        layers.append(("activationOut", output_activation))   

        self.network = torch.nn.Sequential(OrderedDict(layers))  
        self.distribution = distribution


    def forward(self, obs):
        return self.network(obs)
    
    def action(self, obs):
        """
        Samples action and returns action, log p(action)
        """
        parameter = self.forward(obs)

        if self.distribution=="categorical":
            distribution = Categorical(probs=parameter)
        elif self.distribution =="normal":
            distribution = Normal(loc=parameter)
        else:
            distribution = Bernoulli(probs=parameter)

        a = distribution.sample()
        log_p_a = distribution.log_prob(a)

        return a, log_p_a

    def __call__(self, obs):
        return self.action(obs)

