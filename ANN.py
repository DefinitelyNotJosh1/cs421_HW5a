# ANN.py - Artificial Neural Network
# Homework 5a for CS-421
# A very simple artificial neural network implementation in Python. 
# Authors:
# - Joshua Krasnogorov
# - Trenton Pham

import numpy as np
import os

# Define it as a class - will make the implementation in ReANTICS way easier if we do it this way
# Josh:Took inspiration from my own implementation of an ANN in ML class, but simplified it for this assignment
class ANN:
    def __init__(self, input_size, hidden_size, output_size, alpha, batch_size, stop_threshold):
        # Initialize weights and biases
        self.w1 = np.random.rand(input_size, hidden_size) * 2 - 1 # -1 to 1
        self.b1 = np.random.rand(1, hidden_size) * 2 - 1

        self.w2 = np.random.rand(hidden_size, output_size) * 2 - 1 
        self.b2 = np.random.rand(1, output_size) * 2 - 1 

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.accuracy_per_epoch = []
        self.error_per_epoch = [1]
        self.stop_threshold = stop_threshold

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    ##
    # forward
    #
    # Description: Propogates the input forward through the network.
    #
    # Parameters:
    #   input - the input to the network
    # 
    # Return: The output of the network after passing through the network.
    ##
    def forward(self, input):
        # Hidden layer
        self.z1 = np.dot(input, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    ##
    # backward
    #
    # Description: Propogates the error backward through the network.
    #
    # Parameters:
    #   x_input - the input to the network
    #   y_output - the desired output of the network
    #
    # Return: Nothing, updates weights and biases in itself.
    ##
    def backward(self, x_input, y_output):
        # I divide deltas by n to get the average gradient across the batch to make learning rate independent of batch size
        n = x_input.shape[0]
        
        # Output layer error
        # Note: 
        #   for this assignment, omitting the sigmoid derivative makes convergence ~10x faster.
        #   I'll keep it so that it's consistent with what we did in class. May remove for 5b.
        error_output = (y_output - self.a2) * self.sigmoid_derivative(self.a2)

        # Calculate weights and biases for output layer
        dw2 = np.dot(self.a1.T, error_output) / n
        db2 = np.sum(error_output, axis=0, keepdims=True) / n

        # Hidden layer error
        error_hidden = np.dot(error_output, self.w2.T) * self.sigmoid_derivative(self.a1)

        # Calculate weights and biases for hidden layer
        dw1 = np.dot(x_input.T, error_hidden) / n
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / n

        # Update weights and biases
        self.w1 += self.alpha * dw1
        self.b1 += self.alpha * db1
        self.w2 += self.alpha * dw2
        self.b2 += self.alpha * db2

    # Train the model
    def train(self, x_input, y_output):
        epoch = 1
        while (self.error_per_epoch[-1] > self.stop_threshold):
            epoch += 1
            # Shuffle data
            perm = np.random.permutation(len(x_input))
            x_input_shuffled = x_input[perm]
            y_output_shuffled = y_output[perm]

            # Batch training
            batch_count = len(x_input_shuffled) // self.batch_size
            for i in range(batch_count):
                batch_input = x_input_shuffled[i*self.batch_size:(i+1)*self.batch_size]
                batch_output = y_output_shuffled[i*self.batch_size:(i+1)*self.batch_size]

                self.forward(batch_input)
                self.backward(batch_input, batch_output)
            
            output_pred = self.forward(x_input)

            # Calculate accuracy over the full dataset (disabled for this assignment)
            # accuracy = np.mean(output_pred == y_output)
            # self.accuracy_per_epoch.append(accuracy)
            # print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}"

            # Calculate average error over dataset for this epoch
            error = np.mean(np.abs(output_pred - y_output))
            self.error_per_epoch.append(error)

            # print every 100 epochs; wayyyy too many prints if every epoch
            if epoch % 100 == 0: 
                print(f"Epoch {epoch}, Error: {error:.4f}")

            # if average errror is less than stop threshold, stop training
            if error < self.stop_threshold:
                print(f"Training stopped at epoch {epoch+1} at an error of {error:.4f} because average error is less than stop threshold of {self.stop_threshold:.4f}")


# Example training data, this is what we'll use
examples = [
    ([0, 0, 0, 0], [0]),
    ([0, 0, 0, 1], [1]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 1, 1], [1]),
    ([0, 1, 0, 0], [0]),
    ([0, 1, 0, 1], [1]),
    ([0, 1, 1, 0], [0]),
    ([0, 1, 1, 1], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([1, 0, 1, 0], [1]),
    ([1, 0, 1, 1], [1]),
    ([1, 1, 0, 0], [0]),
    ([1, 1, 0, 1], [0]),
    ([1, 1, 1, 0], [0]),
    ([1, 1, 1, 1], [1])
]

# Create an ANN
input_size = 4
output_size = 1
hidden_size = 8
alpha = 0.5
batch_size = 10
stop_threshold = 0.05

ann = ANN(input_size, hidden_size, output_size, alpha, batch_size, stop_threshold)

x_input = np.array([example[0] for example in examples])
y_output = np.array([example[1] for example in examples])

ann.train(x_input, y_output)
