# A module containing the neural network class and its methods

import numpy as np
import scipy.special as scps
from datetime import datetime


# Defining the class for the neural network
# wih: weights input -> hidden
# who: weights hidden -> output
class NeuralNetwork:
    # Initializing
    # Using logistic function as an activation one
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, epochs):
        self.activation_func = lambda x: scps.expit(x)
        self.deactivation_func = lambda x: scps.logit(x)
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.ep = epochs

        self.wih = (np.random.normal(0.0, self.hnodes ** (-0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, self.onodes ** (-0.5), (self.onodes, self.hnodes)))

    # Getting the epochs number
    def get_epochs(self):
        return self.ep

    # Getting the input nodes number
    def get_inodes(self):
        return self.inodes

    # Getting the hidden nodes number
    def get_hnodes(self):
        return self.hnodes

    # Getting the output nodes number
    def get_onodes(self):
        return self.onodes

    # Getting the learning rate number
    def get_lr(self):
        return self.lr

    # The training method
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        # Calculating the error matrix
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # Querying the nodes of the neural network
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

    # Reverse querying
    # input (on the output nodes): transposed 1d array with 0.01 and 0.99 values (0.99 matches the digit)
    # output (on the input nodes): 28*28 elements transposed 1d array with the network results
    # (values from 0.01 to 0.99)
    def reverse_query(self, outputs_list):
        outputs = np.array(outputs_list, ndmin=2).T

        outputs_in = self.deactivation_func(outputs)
        hidden_outputs = np.dot(self.who.T, outputs_in)

        # Next processing operations should be done after a proper scaling
        tmp = np.absolute(hidden_outputs)
        max_val = tmp.max()
        hidden_outputs = hidden_outputs * 0.99 / max_val / 2 + 0.5
        hidden_in = self.deactivation_func(hidden_outputs)
        final_inputs = np.dot(self.wih.T, hidden_in)

        # Final scaling
        tmp = np.absolute(final_inputs)
        max_val = tmp.max()
        final_inputs = final_inputs * 0.99 / max_val / 2 + 0.5

        return final_inputs

    # Configuring weights via preset values
    def set_weights(self, wih_l, who_l):
        if (len(wih_l) == self.hnodes and len(wih_l[0]) == self.inodes and len(who_l) == self.onodes and
                len(who_l[0]) == self.hnodes):
            self.wih = wih_l
            self.who = who_l
        return

    # Passing the weights for export
    def export_weights(self):
        return [self.wih, self.who]

