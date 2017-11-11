import MLP
import numpy as np
import numba
import sklearn as sk
from sklearn.preprocessing import normalize

'''

Creates a data manipulator class to sample, recombine and resample data

'''
class DataHandler:
    def __init__(self, path, folds):
        all_data = sk.preprocessing.maxabs_scale(self.load_data(path))
        self.num_folds = folds
        self.folds_container = []
        self.fold(all_data, folds)
        self.num_features = len(self.folds_container[0].T)-1

    @numba.jit
    def load_data(self, path, token=None):
        return np.loadtxt(path, delimiter=token)  # read file into array


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