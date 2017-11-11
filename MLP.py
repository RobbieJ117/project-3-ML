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
        self.weights_ih = np.random.normal(loc=0.0, scale=.001, size=(in_dim, h_nodes))
        # weights_ho are the weights from the hidden nodes to the output layer
        self.weights_ho = np.random.normal(loc=0.0, scale=.1, size=(h_nodes, out_dim))
        self.epochs = 3 #number of iterations of weight correction steps
        self.ada = -.005
        self.loss_history = [] #used to store error for error_vs_time graphs
        self.batch_size = 1800
        self.iteration=0
        self.product= None
        self.samples_by_nodes = None
        self.out_vector = None


    def feed_forward(self, X):
        #multiply inputs across first set of weights into h_nodes
        self.product = X.dot(self.weights_ih)
        #apply sigmoid function to values in h_nodes
        # this matrix is used also to calculate backpropagated error
        self.samples_by_nodes = np.tanh(self.product)
        # mutiply values in h_nodes across weights to output node
        # out_vector is the vector of outputs from the network
        # its dimensions are (number samples in X)x(1)
        self.out_vector = self.samples_by_nodes.dot(self.weights_ho)
        return self.out_vector

    def backprop(self, X, Y):
        batch_X = np.array
        batch_Y_act = np.array
        batch_Y_pred = np.array
        # if the number of data points can be evenly divided by the batch size
        if (len(Y) % self.batch_size == 0):
            num_batches = len(Y) / self.batch_size  # number of batches given data size
        # if the number of data points in X can not be evenly divided by the batch size
        else:
            num_batches = (len(Y) // self.batch_size) + 1
        # This loop repeats for each batch
        num_batches = (int)(num_batches + 0)
        for i in range(0, num_batches):
            '''
            The below block of code splits the data into smaller batches so that
            gradient descent and weight update can be done in batches
            '''
            for j in range(0, self.epochs):
                print("epoch:{}\n".format(j))
                # if it is the last batch, do not define array ending index
                if (i == (num_batches - 1)):
                    batch_X = self.batch_split(X, i, 0)
                    batch_Y_act = self.batch_split(Y, i, 0)
                    batch_Y_pred = self.feed_forward(batch_X)
                else:
                    batch_X = self.batch_split(X, i, 1)
                    batch_Y_act = self.batch_split(Y, i, 1)
                    batch_Y_pred = self.feed_forward(batch_X)
                error_0 = (batch_Y_act - batch_Y_pred)
                error_1 = -1*((batch_Y_act - np.tanh(batch_Y_pred)))/len(batch_X)
                error_2 = np.multiply(error_0, error_0)
                loss = .5 * np.sum(error_2) / len(batch_X)
                self.loss_history.append(loss)
                if (loss > .01):
                    # The below applies gradient descent for each batch for each epoch
                    # derivative of the hidden layer
                    temp1 = np.ndarray(shape=(self.samples_by_nodes.shape))
                    temp1.fill(1)
                    h_deriv = (temp1 - np.multiply(np.tanh(self.product), np.tanh(self.product)))
                    temp2 = np.ndarray(shape=(self.out_vector.shape))
                    temp2.fill(1)
                    o_deriv = (temp2 - np.multiply((np.tanh(batch_Y_pred)), (np.tanh(batch_Y_pred))))
                    d3 = np.multiply((error_1), o_deriv)
                    dJdW2 = np.dot(self.samples_by_nodes.T, d3)

                    d2 = np.multiply(np.dot(d3, self.weights_ho.T), h_deriv)
                    dJdW1 = np.dot(batch_X.T, d2)
                    self.weights_ih += self.ada * dJdW1
                    self.weights_ho += self.ada * dJdW2
                    self.iteration += 1
                else:
                    j+=1

    def batch_split(self, X, i, switch):
        start = (self.batch_size * i)
        end = ((self.batch_size *(1+ i)))
        if(switch==0):
            temp = X[start:]
            return temp
        else:
            temp = X[start:end]
            return temp

    def test(self, X, Y):
        i = self.iteration-1
        G = self.feed_forward(X)
        #print(Y)
        #print(G)
        mae = metrics.mean_absolute_error(Y, G)
        rmse = metrics.mean_squared_error(Y, G)
        rmse = math.sqrt(rmse)
        mean_y = Y[10:20]
        mean_g = G[10:20]
        #A = np.hstack((X,G)) #set of vectors of predicted points
        #B = np.hstack((X,Y)) #set of vectors for actual points
        #res = 1 - np.dot(A / np.linalg.norm(A, axis=1)[..., None], (B / np.linalg.norm(B, axis=1)[..., None]).T)# compute cosine distance between vectors
        #cos_dist = res.mean()# mean cosine distance
        cos_dist = "\t Not currently implemented"
        reults_string = "\nIteration{}\n\nRMSE:{}\nMAE:{}\nMean Cosine similarity{}\nMean Y{}\n\n Mean G{}\n\n".format(i, rmse, mae, cos_dist, mean_y, mean_g)
        if not os.path.isfile(self.id):
            f = open(self.id, "w")
            header = "{}:\nAda:{}\nEpochs:{}\nhidden nodes{}\nBatch size: {}\n".format(self.id, self.ada, self.epochs, self.h_nodes, self.batch_size)
            f.write(header)
        else:
            f = open(self.id, "a")
        f.write(reults_string)
        f.close()


    '''
    Prints a plot of the error versus training iterations
    '''

    def print_results(self):
        file_name = "{}.png".format(self.id)
        fig = plt.figure()
        plt.plot(np.arange(0, len(self.loss_history)), self.loss_history)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        fig.savefig(file_name)
        plt.close(fig)
        self.lossHistory = []