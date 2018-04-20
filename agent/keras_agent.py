import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])
actions_space = env.action_space.n




