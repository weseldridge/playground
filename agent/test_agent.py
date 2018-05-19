import os
import sys
import numpy as np

import pommerman
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman.constants import GameType

print(pommerman.registry)


# Create a set of agents (exactly four)
agent_list = [
    SimpleAgent(),
    RandomAgent(),
    SimpleAgent(),
    RandomAgent(),
]

env = pommerman.make('PommeFFA-v0', agent_list)


for i_episode in range(10):
    state = env.reset()
    done = False
    while not done:
        env.render()
        actions = env.act(state)
        state, reward, done, info = env.step(actions)
        if done:
            print(info)
    print('Episode {} finished'.format(i_episode))