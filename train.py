import gym
from learning.agent import Agent
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

env = gym.make('LunarLander-v2')
env.seed(0)

agent_config = {
    "replay_buffer_config": {
        "buffer_size": 100000,
        "minibatch_size": 32,
        "seed": 0
    },
    "network_config": {
        "state_dimention": 8,
        "action_dimention": 4,
        "hidden_layers": [256, 256],
    },
    "num_replay_per_step": 16,
    "gamma": 0.99,
    "tau": 0.01,
    "seed": 0
}
agent = Agent()
agent.init(agent_config)
action = agent.start(env.reset())

while True:
    env.render()
    observation, reward, done, info = env.step(action)
    if done:
        agent.end(reward)
        break
    action = agent.step(reward, observation)