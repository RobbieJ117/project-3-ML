import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
import os.path
import numba

class MLP(object):

    def __init__(self, experiment_id, in_dim, h_nodes, out_dim):
        self.id = "{}.txt".format(experiment_id)
        self.in_dim = in_dim #dimensions/features of the input data
        self.h_nodes = h_nodes #number of hidden nodes
        self.out_dim = out_dim # number of output nodes
        # weights_ih are the weights from the input data vectors to the hidden layer
        self.weights_ih = np.random.uniform(low=-.1, high = .1, size=(in_dim, h_nodes))
        # weights_ho are the weights from the hidden nodes to the output layer
        self.weights_ho = np.random.uniform(lpw=-.1, high=.1, size=(h_nodes, out_dim))
        self.epochs = 20 #number of iterations of weight correction steps
        self.ada = -.1
        self.loss_history = [] #used to store error for error_vs_time graphs


    def feed_forward(self, X):
        #multiply inputs across first set of weights into h_nodes
        product = X.dot(self.weights_ih)
        #apply sigmoid function to values in h_nodes
        # this matrix is used also to calculate backpropagated error
        self.samples_by_nodes = np.tanh(product)
        # mutiply values in h_nodes across weights to output node
        # out_vector is the vector of outputs from the network
        # its dimensions are (number samples in X)x(1)
        out_vector = product.dot(self.weights_ho)
        return out_vector






    def gradient_descent(self, X, Y, batch_size):
        '''
                The below block of code splits the data into smaller batches so that
                gradient descent and weight update can be done in batches
                '''
        # if the number of data points can be evenly divided by the batch size

        if (len(Y) % batch_size == 0):
            num_batches = len(Y) / batch_size  # number of batches given data size
        # if the number of data points in X can not be evenly divided by the batch size
        else:
            num_batches = (len(Y) // batch_size) + 1
        for i in range(0, num_batches):
            # if it is the last batch, do not define array ending index

            for j in range(0, self.epochs):
                if (i == (num_batches - 1)):
                    batch_Y_act = Y[(batch_size * i):]
                    batch_Y_pred = self.feed_forward(X[(batch_size * i):])
                else:
                    batch_Y_act = Y[(batch_size * i):((batch_size + 1) * i)]
                    batch_Y_pred = self.feed_forward(X[(batch_size * i):((batch_size + 1) * i)])
                error = metrics.mean_squared_error(batch_Y_act, batch_Y_pred)
                loss = np.sum(error)
                if(loss<.005):
                    break
                else:
                    self.loss_history.append(loss)
                    gradient_ho = self.samples_by_nodes.T.dot(error)
                    gradient_ho = (gradient_ho*self.ada)/len(batch_Y_act)
                    self.weights_ho += gradient_ho


