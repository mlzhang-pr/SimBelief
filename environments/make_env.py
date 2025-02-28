"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import os

import gym
# import gymnasium
import numpy as np
import torch
from gym.spaces.box import Box

from environments.wrappers import VariBadWrapper#,VariBadWrapper_minigrid
# from minigrid.wrappers import FlatObsWrapper

def make_env(env_id, episodes_per_task, seed=None, **kwargs):
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.seed(seed)
    env = VariBadWrapper(env=env,
                         episodes_per_task=episodes_per_task,
                         env_name=env_id
                         )
    return env

# def make_env_minigrid(env_id, episodes_per_task, seed=None, **kwargs):
#     env = gymnasium.make(env_id, **kwargs)
#     if seed is not None:
#         env.seed(seed)
#     env =FlatObsWrapper(env)
#     env = VariBadWrapper_minigrid(env=env,
#                          episodes_per_task=episodes_per_task,
#                          )
#     return env
