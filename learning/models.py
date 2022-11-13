from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization

class ActionValue(Model):
    def __init__(self, state_dimention, action_dimention, hidden_layers):
        super().__init__()
        self.state_dimention = state_dimention
        self.action_dimention = action_dimention
        self.hidden_layers = hidden_layers
        
        self.input_layer = Dense(self.state_dimention, activation='relu')
        self.batch_norm = BatchNormalization()
        self.hidden_layers = [Dense(layer_size, activation='relu') for layer_size in self.hidden_layers]
        self.batch_norms = [BatchNormalization() for _ in self.hidden_layers]
        self.output_layer = Dense(self.action_dimention, activation='linear')

    def call(self, state):
        x = self.input_layer(state)
        x = self.batch_norm(x)
        for layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            x = layer(x)
            x = batch_norm(x)
        return self.output_layer(x)