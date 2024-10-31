import torch
import gymnasium
from torch.optim import Adam

from src.policies.policy import Policy
from src.values.value import Value
from src.sampling.sampler import Sampler
from src.functions.helpers import return_2g, average_return, values, advantage, average_len
from src.functions.policy_loss import advantage_weighted_log_prob
from src.functions.value_loss import mc_value_loss



env = gymnasium.make("MountainCar-v0")
# env._max_episode_steps = 500

policy_fn = Policy(n_hidden=5, n_nodes=20, input_dim=2, output_dim=3, distribution="categorical")
value_fn = Value(n_hidden=5, n_nodes=20, input_dim=2, output_dim=1)


sampler = Sampler(policy=policy_fn, env=env)

p_opt = Adam(policy_fn.parameters(), lr=0.001)
v_opt = Adam(value_fn.parameters(), lr=0.001)

epochs = 500
episodes = 100

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
        f"\tAverage return: {round(average_return(rewards).item(), 2)}"
        f"\tAverage length: {round(average_len(rewards), 2)}")
