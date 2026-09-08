import numpy as np

class layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.biases    
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class NeuralNetwork:
    def __init__(self):
        self.dense1 = layer_Dense(6, 5)
        self.activation1 = Activation_ReLU()
        self.dense2 = layer_Dense(5, 4)
        self.activation2 = Activation_ReLU()
        self.dense3 = layer_Dense(4, 3)
    def forward(self, inputs):
        self.dense1.forward(inputs)
        self.activation1.forward(self.dense1.output)

        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)

        self.dense3.forward(self.activation2.output)
        return self.dense3.output