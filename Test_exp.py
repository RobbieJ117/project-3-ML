import MLP
import numpy as np
import numba
from sklearn.preprocessing import normalize

'''

Creates a data manipulator class to sample, recombine and resample data

'''
class Data_manipulator:
    def __init__(self, path, folds):
        all_data = self.load_data(path)
        print(all_data[1])
        self.num_folds = folds
        #normalize(all_data)
        self.folds_container = []
        self.fold(all_data, folds)
        self.num_features = len(self.folds_container[0].T)-1

    @numba.jit
    def load_data(self, path):
        return np.loadtxt(path, delimiter=",")  # read file into array


    def fold(self, input_data, folds):
        self.num_folds = folds
        fold_end_index = len(input_data)//self.num_folds
        if(len(input_data)%self.num_folds ==0):
            fold_end_index -= 1
        for i in range(0, self.num_folds):
            if(i==self.num_folds-1):
                fold_i = input_data[(i*fold_end_index):]
                self.folds_container.append(fold_i)
            else:
                fold_i = input_data[(i*fold_end_index):((i+1)*fold_end_index)]
                self.folds_container.append(fold_i)

    def cleave(self, in_matrix):
        x = in_matrix[:,0:-1]
        y = in_matrix[:,-1:]
        return x, y

    def recombine(self, num_folds):
        temp_array = self.folds_container[0]
        for i in range(0, self.num_folds-1):
            temp_array = np.vstack((temp_array, self.folds_container[i+1]))
        np.random.shuffle(temp_array)
        self.folds_container.clear()
        self.fold(temp_array, num_folds)

class Experimenter:
    def __init__(self, dm, mlp, training_method):
        self.dm = dm
        self.mlp = mlp
        self.train_method = training_method
        self.populatoin = []

    def train(self, features, targets):
        if (self.train_method==1):
            self.mlp.backprop(features, targets)

    def test(self, features, targets):
        if (self.train_method==1):
            self.mlp.test(features, targets)

    def five_by_two(self):
        for i in range(0, 5):
            self.dm.recombine(2)
            x1, y1 = dm.cleave(dm.folds_container[0])
            self.train(x1, y1)
            x2, y2 = dm.cleave(dm.folds_container[1])
            self.test(x2, y2)
            print("\nIteration {} complete".format(i))
        self.mlp.print_results()

dm = Data_manipulator("3D_test2.txt", 1)
network = MLP.MLP("new format test", dm.num_features, 6, 1)
exp_1 = Experimenter(dm, network, 1)
exp_1.five_by_two()
