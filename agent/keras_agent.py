import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam 

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import pommerman
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman import agents

ENV_NAME = "Pomme"

# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])
nb_actions = env.action_space.n


class KerasDDQNAgent(DQNAgent, BaseAgent):
    def __init__(self, model, policy=None, test_policy=None, 
                 enable_double_dqn=True, enable_dueling_network=False,
                 nb_steps_warmup=1000,
                 target_model_update=1000,
                 dueling_type='avg', nb_actions=None, memory=None, *args, **kwargs):
        DQNAgent.__init__(self, model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=nb_steps_warmup,
               enable_dueling_network=enable_dueling_network, 
               dueling_type=dueling_type, 
               target_model_update=target_model_update, 
               policy=policy)
        BaseAgent.__init__(self)

        # super(DQNAgent, self).__init__(model=model, policy=policy, test_policy=test_policy, 
        #          enable_double_dqn=enable_double_dqn, enable_dueling_network=enable_dueling_network,
        #          dueling_type=dueling_type, nb_actions=nb_actions, memory=memory)

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
print((1,) + env.observation_space.shape)
exit()
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = KerasDDQNAgent(model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=10,
               enable_dueling_network=True, 
               dueling_type='avg', 
               target_model_update=1e-2, 
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

agent_list = [
    agents.SimpleAgent(),
    agents.RandomAgent(),
    agents.SimpleAgent(),
    dqn
]

env = pommerman.make('PommeFFA-v0', agent_list)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)