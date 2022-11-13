from learning.memory import ReplayBuffer
from learning.models import ActionValue
import numpy as np
from learning.utils import *
from copy import deepcopy

class Agent():
    def __init__(self, agent_config):
        self.replay_buffer = ReplayBuffer(**agent_config["replay_buffer_config"])
        self.network = ActionValue(**agent_config['network_config'])
        self.network.compile(optimizer='adam', loss=td_error)
        self.num_actions = agent_config['network_config']['action_dimention']
        self.num_replay = agent_config['num_replay_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        self.state = None
        self.action = None
        self.sum_rewards = 0
        self.episode_steps = 0

    def pick_action(self, state):
        action_values = self.network(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    def start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0
        self.state = np.array([state])
        self.action = self.pick_action(self.state)
        return self.action

    def step(self, reward, next_state):
        self.sum_rewards += reward
        self.episode_steps += 1
        next_state = np.array([next_state])
        self.replay_buffer.append(self.state, self.action, reward, False, next_state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.learn()
        self.state = next_state
        self.action = self.pick_action(next_state)
        return self.action

    def learn(self):
        for _ in range(self.num_replay):
            experiences = self.replay_buffer.sample()
            states, actions, rewards, terminals, next_states = map(np.array, zip(*experiences))
            states = states.reshape(len(experiences), -1)
            next_states = next_states.reshape(len(experiences), -1)
            target_q = get_target_q(states, 
                                    next_states, 
                                    actions, 
                                    rewards, 
                                    self.discount, 
                                    terminals, 
                                    self.network, 
                                    self.tau)
            batch_size = self.replay_buffer.minibatch_size
            target_q_actions = np.array(self.network(states))
            target_q_actions[range(batch_size), actions] = target_q
            self.network.train_on_batch(states, target_q_actions)

    def end(self, reward):
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.zeros_like(self.state)
        self.replay_buffer.append(self.state, self.action, reward, True, state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.learn()
        
    def message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")