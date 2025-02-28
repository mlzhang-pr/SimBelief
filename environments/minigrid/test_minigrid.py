# env = gym.make('MiniGrid-Empty-8x8-v0')
# obs1 = env.reset()  # obs: {'image': numpy.ndarray (7, 7, 3),'direction': ,'mission': ,}
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# obs2 = env.reset()  # obs: {'mission': ,'image': numpy.ndarray (56, 56, 3)}
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
# obs3 = env.reset()  # obs: numpy.ndarray (56, 56, 3)

# # 不能在使用上述Wrapper后再使用此FlatObsWrapper，应该单独使用
# env = gym.make('MiniGrid-Empty-8x8-v0')
# env = FlatObsWrapper(env)
# obs4 = env.reset()  # obs: numpy.ndarray  (56, 56, 3)

import minigrid
# import gymnasium as gym
import gymnasium as gym
import matplotlib.pyplot as plt

from minigrid.wrappers import DictObservationSpaceWrapper
from minigrid.wrappers import FlatObsWrapper
env = gym.make("MiniGrid-LavaCrossingS11N5-v0")

obs, _ = env.reset()

print(obs['mission'])
'avoid the lava and get to the green goal square'

env_obs = DictObservationSpaceWrapper(env)

obs, _ = env_obs.reset()

print(obs['mission'][:10])












