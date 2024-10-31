import torch

def return_weighted_log_prob(batch_log_probs, batch_return):
    """
    sums log_p * total_return for each episode and then returns the negative average of these.
    """
    losses = []
    for e in range(len(batch_log_probs)):
        # make sure the return is detached
        ret = batch_return[e].detach()
        losses.append((batch_log_probs[e] * ret).sum())

    return - torch.stack(losses).mean()

def return_2g_weighted_log_prob(batch_log_probs, batch_ret_2g):
    """
    sums log_p * return to go for each episode and then returns the negative average of these.
    """
    losses = []
    for e in range(len(batch_log_probs)):
        ret_2g = batch_ret_2g[e].detach()
        losses.append((batch_log_probs[e] * ret_2g).sum())

    return - torch.stack(losses).mean()

def advantage_weighted_log_prob(batch_log_probs, batch_advantages):
    """
    sums log_p * advantage for each episode and then returns the negative average of these.
    """
    losses = []
    for e in range(len(batch_log_probs)):
        adv = batch_advantages[e].detach()
        losses.append((batch_log_probs[e] * adv).sum())

    return - torch.stack(losses).mean()