import torch
import numpy as np

class Sampler:

    """
    Base class for sampling episodes.
    """

    def __init__(self, env, policy, action_mapping= lambda x: int(x.item())):
        self.env = env
        self.policy = policy

        # use a lambda for customisable mapping from torch tensor to environment.step() input
        self.action_mapping = action_mapping

    def sample_episode(self):
        actions = []
        log_probs = []
        observations = []
        rewards = []
        done = False
        truncated = False
        obs, _ = self.env.reset()
        observations.append(obs)

        while not done and not truncated:

            # sample action
            obs_tensor = torch.tensor(obs, requires_grad=False, dtype=torch.float32)
            action, log_p = self.policy(obs_tensor)
            a = self.action_mapping(action)

            # take step
            obs, rew, done, truncated, _ = self.env.step(a)

            # append step data
            actions.append(a)
            log_probs.append(log_p)
            observations.append(obs)
            rewards.append(rew)
        
        # data formatting
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
        log_probs_t = torch.stack(log_probs)
        observations_t = torch.tensor(np.array(observations[:-1]), dtype=torch.float32)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)

        return actions_t, log_probs_t, observations_t, rewards_t


    def sample_batch(self, n_episodes):
        """
        Sample n episodes
        """
        batch_actions = []
        batch_log_probs = []
        batch_observations = []
        batch_rewards = []

        for episode in range(n_episodes):

            actions, log_probs, observations, rewards = self.sample_episode()

            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_observations.append(observations)
            batch_rewards.append(rewards)
        
        return batch_actions, batch_log_probs, batch_observations, batch_rewards
        
    
    