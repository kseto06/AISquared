# import gymnasium as gym
# import numpy as np

# class CustomEnv(gym.Env):
#     """Custom Environment that follows gym interface."""

#     metadata = {"render_modes": ["human"], "render_fps": 30}

#     def __init__(self, arg1, arg2):
#         pass

#         super().__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
#         # Example for using image as input (channel-first; channel-last also works):
#         self.observation_space = gym.spaces.Box(low=0, high=255,
#                                             shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

#     def step(self, action):
#         pass
#         return observation, reward, terminated, truncated, info

#     def reset(self, seed=None, options=None):
#         pass
#         return observation, info

#     def render(self):
#         pass

#     def close(self):
#         pass


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

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
        self.hidden_2 = nn.Linear(in_features=128, out_features=32)
        # ReLU here
        self.output = nn.Linear(in_features=32, out_features=features_dim)        
    
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
                            features_extractor_kwargs=dict(features_dim=128))

if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    model = PPO(policy=CustomPolicy, env=env, verbose=1)
    model.learn(total_timesteps=10000)
    



    