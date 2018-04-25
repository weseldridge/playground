import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym


import pommerman
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman import agents
from pommerman import helpers
from pommerman import make

record_json_dir="/Users/weseldridge/Developer/Pommerman/game_state"
record_pngs_dir="/Users/weseldridge/Developer/Pommerman/train_images"
game_state_file=None#"/Users/weseldridge/Developer/Pommerman/playground/agent/start.json"

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class DQNTensorForceAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, character=Bomber):
        super(DQNTensorForceAgent, self).__init__(character)

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        return None

    def initialize(self, env):
        from gym import spaces
        from tensorforce.agents import DQNAgent

        
        if type(env.action_space) == spaces.Tuple:
            actions = {str(num): {'type': int, 'num_actions': space.n}
                        for num, space in enumerate(env.action_space.spaces)}
        else:
            actions = dict(type='int', num_actions=env.action_space.n)

        return DQNAgent(
            states=dict(type='float', shape=env.observation_space.shape),
            actions=actions,
            network=[
                dict(type='dense', size=64),
                dict(type='dense', size=64)
            ],
            saver=dict(
                directory="/Users/weseldridge/Developer/Pommerman/models/dqn2",
                seconds=300
            ),
            discount=0.7,
            batching_capacity=1000,
            actions_exploration=dict(
                type="epsilon_decay",
                initial_epsilon=1.0,
                final_epsilon=0.1,
                timesteps=500
            )
        )

class WrappedEnv(OpenAIGym):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, actions):
        if self.visualize:
            self.gym.render(record_json_dir=record_json_dir, record_pngs_dir=record_pngs_dir)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs


config="PommeFFA-v0"

agents = [
    agents.SimpleAgent(),
    agents.RandomAgent(),
    agents.SimpleAgent(),
    DQNTensorForceAgent()
]




env = make(config, agents, game_state_file=game_state_file)
training_agent = None

for agent in agents:
    if type(agent) == DQNTensorForceAgent:
        training_agent = agent
        env.set_training_agent(agent.agent_id)
        print("Agent id:" + str(agent.agent_id))
        break

agent = training_agent.initialize(env)

atexit.register(functools.partial(clean_up_agents, agents))

def episode_finished(r):
       print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))
       return True

wrapped_env = WrappedEnv(env, visualize=True)
runner = Runner(agent=agent, 
                environment=wrapped_env)
runner.run(episodes=1, max_episode_timesteps=2000, episode_finished=episode_finished)

runner.close()