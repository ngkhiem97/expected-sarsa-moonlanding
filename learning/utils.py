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
    actions_q_next = network(next_states)
    probs_mat = softmax(actions_q_next, tau=tau)
    expected_q_next = np.sum(probs_mat * actions_q_next, axis=1)*(1 - terminals)
    expected_q_next.squeeze()
    target_q = rewards + discount * expected_q_next
    target_q = target_q.squeeze()
    return target_q

def td_error(y, y_hat):
    return y_hat - y
