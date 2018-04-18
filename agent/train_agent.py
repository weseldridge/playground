import os
import sys
import numpy as np

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


# Instantiate the environment
config = ffa_v0_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states=dict(type='float', shape=env.observation_space.shape),
    actions=dict(type='int', num_actions=env.action_space.n),
    network=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batching_capacity=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

# Add 3 random agents
agents = []
for agent_id in range(3):
    agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent
agent_id += 1
agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
env.set_agents(agents)
env.set_training_agent(agents[-1].agent_id)
env.set_init_game_state(None)

class WrappedEnv(OpenAIGym):    
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
    
    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward
    
    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs

# Instantiate and run the environment for 5 episodes.
wrapped_env = WrappedEnv(env, True)
runner = Runner(agent=agent, environment=wrapped_env)
runner.run(episodes=5, max_episode_timesteps=2000)
print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)

try:
    runner.close()
except AttributeError as e:
    pass