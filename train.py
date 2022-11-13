import gym
from learning.agent import Agent
import numpy as np
import tensorflow as tf

EPOCH = 1000

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

env = gym.make('LunarLander-v2')
env.seed(0)

agent_config = {
    "replay_buffer_config": {
        "buffer_size": 500,
        "minibatch_size": 32,
        "seed": 0
    },
    "network_config": {
        "state_dimention": 8,
        "action_dimention": 4,
        "hidden_layers": [256, 256],
    },
    "num_replay_per_step": 4,
    "gamma": 0.99,
    "tau": 0.05,
    "seed": 0
}
agent = Agent(agent_config)
action = agent.start(env.reset())

epoch = 0
while True:
    if epoch % 50 == 0:
        agent.network.save_weights(f'./models/epoch_{epoch}.h5')
    observation, reward, done, info = env.step(action)
    if done:
        agent.end(reward)
        print("Epoch: {}, Reward: {}, Steps: {}".format(epoch, agent.sum_rewards, agent.episode_steps))
        action = agent.start(env.reset())
        epoch += 1
        continue
    action = agent.step(reward, observation)