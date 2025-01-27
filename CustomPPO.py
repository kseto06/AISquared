import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
# from pettingzoo.butterfly import cooperative_pong_v5
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

class MultiAgentPPO(gym.Env):
    '''
    Define the class for the custom multi-agent PPO 
    Use LSTM + PPO NN for self-play
    '''
    
    pass

class CustomNN(BaseFeaturesExtractor):
    def __init__(self, obs_space: gym.spaces.Box, features_dim: int):
        super().__init__(obs_space, features_dim)
        # Layers
        self.input = nn.Linear(in_features=obs_space.shape[0], out_features=obs_space.shape[0]) #input, output
        self.LSTM = nn.LSTM(input_size=obs_space.shape[0], hidden_size=512, num_layers=1, batch_first=True)
        self.hidden_1 = nn.Linear(in_features=512, out_features=128)
        self.relu = nn.ReLU()
        self.hidden_2 = nn.Linear(in_features=128, out_features=64)
        # ReLU here
        self.output = nn.Linear(in_features=64, out_features=features_dim)        
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input(observations)

        # Adjust to 3D Tensor for LSTM if currently 2D
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)

        x, _ = self.LSTM(x)
        x = self.hidden_1(x.squeeze(1)) #Use last hidden state of LSTM
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            features_extractor_class=CustomNN,
                            features_extractor_kwargs=dict(features_dim=64)) #Default features_dim for MlpPolicy is 64

# class PettingZooGymWrapper(gym.Env):
#     def __init__(self, env):
#         super(PettingZooGymWrapper, self).__init__()
#         self.env = env
#         self.agents = env.agents
#         self.current_agent = None
        
#         # Define action and observation space
#         self.action_space = gym.spaces.Discrete(env.action_space(self.agents[0]).n)  # Assuming discrete action space
#         self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(env.observation_space(self.agents[0]).shape,), dtype=float)

#     def reset(self, seed=None, options=None):
#         self.env.reset()
#         self.current_agent = self.env.agents[0]  # Start with the first agent
#         obs = self.env.observe(self.current_agent)
#         return obs, {}

#     def step(self, action):
#         self.env.step(action)
#         self.current_agent = self.env.agent_next()
#         obs = self.env.observe(self.current_agent)
#         reward = self.env.rewards[self.current_agent]
#         done = self.env.dones[self.current_agent]
#         truncated = self.env.truncations[self.current_agent]
#         return obs, reward, done, truncated, {}

#     def render(self):
#         self.env.render()

#     def close(self):
#         self.env.close()

if __name__ == '__main__':
    env = gym.make("LunarLander-v3")

    model = PPO(policy=CustomPolicy, env=env, verbose=1)
    model.learn(total_timesteps=100000)

    # env = cooperative_pong_v5.env(render_mode="human")
    # env.reset(seed=42) 
    # env = PettingZooGymWrapper(env)
    
    # env = gym.vector.make(lambda: env, num_envs=1)
    # env = DummyVecEnv([lambda: env])

    # agents = ['agent1', 'agent2']
    # models = {agent: PPO(policy=CustomPolicy, env=env, verbose=1) for agent in agents}

    # for agent, model in models.items():
    #     model.learn(total_timesteps=100000)

    obs = env.reset()
    done = False
    while not done:
        actions, states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(actions)
        env.render()
        done = any(dones)
    



    