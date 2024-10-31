import torch

def average_return(rewards):
    """
    Gives the average return per episode of an entire batch.
    """
    rets = []
    for rew in rewards:
        rets.append(rew.sum())
    
    return torch.stack(rets).mean()


def average_len(rewards):
    """
    Gives the average length per episode of an entire batch.
    """
    lens = 0
    for rew in rewards:
        lens += len(rew)
    
    return lens/len(rewards)


def return_2g(rewards):
    """
    Takes a batch of rewards and calculates the return to go.

    Is a reverse cumulative sum. 
    """
    cumsum_rews = []
    for rew in rewards:
        cumsum_rews.append(rew + torch.sum(rew) - torch.cumsum(rew, 0))
    return cumsum_rews


def values(value_function, observations):
    """
    Get the values for a batch of observations
    """
    values = []
    for obs in observations:
        value = value_function(obs)
        values.append(value)
    
    return values


def advantage_targets(values, rewards):
    """
    r_t + value(s_t+1). 
    """
    targets = []
    for vals, rew in zip(values, rewards):
        next_vals = torch.cat((vals[1:].clone(), torch.tensor([0], dtype=torch.float32)))
        targets.append(rew + next_vals)

    return targets


def advantage(values, rewards):
    """
    Returns the advantage of a given state and action.
    
    advantage(s_t, a_t) = r_t + value(s_t+1) - value(s_t)
    """
    advantages = []
    for e in range(len(values)):
        next_vals = torch.cat((values[e][1:].clone(), torch.tensor([0], dtype=torch.float32)))
        advantages.append(rewards[e] + next_vals - values[e])

    return advantages