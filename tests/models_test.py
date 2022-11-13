from learning import models
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np

network_config = {
    'state_dimention': 4,
    'action_dimention': 2,
    'hidden_layers': [10, 10]
}

input = np.random.rand(1, 4)
action_value = models.ActionValue(network_config)
output = action_value.apply(input)

print(input)
print(output)