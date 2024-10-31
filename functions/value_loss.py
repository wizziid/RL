import torch

def mc_value_loss(reward_to_go, values):
    """
    returns the MSE of values computed by the value function and the associated rewards to go.
    """

    losses = []
    for r, v in zip(reward_to_go, values):
        losses.append(torch.nn.functional.mse_loss(r, v))
    
    return torch.stack(losses).mean()


def target_value_loss(values, targets):
    """
    returns the MSE of values computed by the value function and the associated targets (r_t + v(s_t+1)).

    Exactly the same as above but names differently to be more explicit...
    """

    losses = []
    for r, v in zip(targets, values):
        losses.append(torch.nn.functional.mse_loss(r, v))
    
    return torch.stack(losses).mean()