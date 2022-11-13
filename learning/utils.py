import numpy as np
import tensorflow as tf

def softmax(action_values, tau=1.0):
    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1)
    reshaped_max_preference = max_preference.reshape((-1, 1))
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    action_probs = action_probs.squeeze()
    return action_probs

def get_target_q(states, next_states, actions, rewards, discount, terminals, network, tau):
    # conversion to numpy arrays for calculations
    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)[: , np.newaxis]
    terminals = np.array(terminals)[: , np.newaxis]

    actions_q_next = network(next_states)
    probs_mat = np.array([softmax(actions_q_next[i], tau=tau)[np.newaxis, :] for i in range(actions_q_next.shape[0])])
    expected_q_next = np.sum(probs_mat * actions_q_next, axis=2)*(1 - terminals)
    expected_q_next.squeeze()
    target_q = rewards + discount * expected_q_next
    target_q = target_q.squeeze()
    return target_q

def td_error(y, y_hat):
    return tf.where(y_hat != 0, y_hat - y, 0)
