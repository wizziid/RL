import torch
import gymnasium
from torch.optim import Adam

from policies.policy import Policy
from values.value import Value
from sampling.sampler import Sampler
from functions.helpers import return_2g, average_return, values, advantage
from functions.policy_loss import advantage_weighted_log_prob
from functions.value_loss import mc_value_loss


def main():
    
    policy_fn = Policy(n_hidden=5, n_nodes=20, input_dim=4, output_dim=1, distribution="bernoulli")
    value_fn = Value(n_hidden=5, n_nodes=20, input_dim=4, output_dim=1)

    env = gymnasium.make("CartPole-v1")

    sampler = Sampler(policy=policy_fn, env=env)

    p_opt = Adam(policy_fn.parameters(), lr=0.01)
    v_opt = Adam(value_fn.parameters(), lr=0.01)

    epochs = 100
    episodes = 500

    for epoch in range(epochs):

        actions, log_probs, observations, rewards = sampler.sample_batch(n_episodes=episodes)

        ret_2g = return_2g(rewards)

        vals = values(value_fn, observations)

        advs = advantage(values=vals, rewards=rewards)

        p_loss = advantage_weighted_log_prob(batch_log_probs=log_probs, batch_advantages=advs)
        p_opt.zero_grad()
        p_loss.backward()
        p_opt.step()

        v_loss = mc_value_loss(reward_to_go=ret_2g, values=vals)
        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        print(f"Epoch: {epoch+1}/{epochs} "
            f"\tPolicy loss: {round(p_loss.item(), 2)} "
            f"\tValue loss: {round(v_loss.item(), 2)} "
            f"\tAverage return: {round(average_return(rewards).item(), 2)}")


if __name__ == "__main__":
    main()