import numpy as np
import random

class Network:
    #layers, biases, weights
    def __init__(self, size):
        self.nr_layers = len(size)
        self.size = size
        self.bias = [np.random.rand(y, 1) for y in size[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(size[1:], size[:-1])]

    def feedfoward(self, a):
        #a is activation of last layer(or input)
        for b,w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return(a)

    def SGD(self, training_data, test_data, nr_epoch, mini_batch_size, learning_rate):
        n_test_data = len(test_data)
        n_training_data = len(training_data)
        #build mini batches
        for i in range(nr_epoch):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j + mini_batch_size]
                            for j in range(0,n_training_data,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            print("Epoch {} : {} / {}".format(i, self.evaluate(test_data), n_test_data))

    def update_mini_batch(self, mini_batch, learning_rate):
        bias_gradient = [np.zeros(bias.shape) for bias in self.bias]
        weights_gradient = [np.zeros(weights.shape) for weights in self.weights]
        #summing up gradients for weights and biases(calculate each gradient with backprop)
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            bias_gradient = [b + db for b, db in zip(bias_gradient, delta_b)]
            weights_gradient = [w + db for w, db in zip(weights_gradient, delta_w)]
        #now we update original weights and biases with gradient descent formula
        self.bias = [b - (learning_rate/len(mini_batch)) * change
                     for b, change in zip(self.bias, bias_gradient)]
        self.weights = [w - (learning_rate/len(mini_batch)) * change
                        for w, change in zip(self.weights, weights_gradient)]

    def backprop(self, x, y):
        bias_gradient = [np.zeros(bias.shape) for bias in self.bias]
        weights_gradient = [np.zeros(weights.shape) for weights in self.weights]
        activation = x
        activations = [x]
        #zs are weighted inputs
        zs = []
        #FEEDFOWARD
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #BACKWARD PASS
        #first last layer(backprop formula #1), then we assign BP3 and BP4
        delta = self.last_layer_cost(activations[-1], y) * sigmoid_derivative(zs[-1])
        bias_gradient = delta
        weights_gradient = np.dot(delta, activations[-2].transpose())
        #now we apply BP formula #2 to all others(l-2) layers, then we assign BP3 and BP4
        #first layer in this loop is last layer before output(a^L)
        for l in range(2, self.nr_layers):
            delta = np.dot(self.weights[-l].transpose(), delta) * sigmoid_derivative(zs[-l])
            bias_gradient = delta
            weights_gradient = np.dot(delta, activations[-l - 1].transpose())
            return weights_gradient, weights_gradient

    def last_layer_cost(self, last_layer_activation, y):
        return(last_layer_activation - y)

    def evaluation(self, test_data):
        test_result = [(np.argmax(self.feedfoward(x), y)) for x, y in test_data]
        return sum(int(x==y) for x, y in test_result)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
