from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class ActionValue(Model):
    def __init__(self, state_dimention, action_dimention, hidden_layers):
        super().__init__()
        self.state_dimention = state_dimention
        self.action_dimention = action_dimention
        self.hidden_layers = hidden_layers
        
        self.input_layer = Dense(self.state_dimention, activation='relu')
        self.hidden_layers = [Dense(layer_size, activation='relu') for layer_size in self.hidden_layers]
        self.output_layer = Dense(self.action_dimention, activation='linear')

    def call(self, state):
        x = self.input_layer(state)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)