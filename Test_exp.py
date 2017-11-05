import MLP
import numpy as np
import numba
from sklearn.preprocessing import normalize

'''

Creates a data manipulator class to sample, recombine and resample data

'''
class Data_manipulator:
    def __init__(self, path):
        all_data = self.load_data(path)
        print(all_data[1])
        self.n = len(all_data)
        self.h =  self.n//2
        normalize(all_data, )
        fold_1, fold_2 = self.shuffle_fold(all_data)
        self.fold_1_x, self.fold_1_y = self.cleave(fold_1)
        self.fold_2_x, self.fold_2_y = self.cleave(fold_2)
    @numba.jit
    def load_data(self, path):
        return np.loadtxt(path, delimiter=",")  # read file into array

    def print_validate(self):
        print("\nF1X\n")
        print(self.fold_1_x)
        print("\nF1Y\n")
        print(self.fold_1_y)
        print("\nF2X\n")
        print(self.fold_2_x)
        print("\nF2Y\n")
        print(self.fold_2_y)

    def shuffle_fold(self, input_data):
        np.random.shuffle(input_data)
        f_1 = input_data[0:self.h, :]
        f_2 = input_data[self.h:-1, :]
        return f_1, f_2

    def cleave(self, in_matrix):
        x = in_matrix[:,0:-1]
        y = in_matrix[:,-1:]
        return x, y

    def recombine(self):
        fold_1 = np.hstack((self.fold_1_x, self.fold_1_y))
        fold_2 = np.hstack((self.fold_2_x, self.fold_2_y))
        whole_data = np.vstack((fold_1, fold_2))
        fold_1, fold_2 = self.shuffle_fold(whole_data)
        self.fold_1_x, self.fold_1_y = self.cleave(fold_1)
        self.fold_2_x, self.fold_2_y = self.cleave(fold_2)

dm = Data_manipulator("3D_test.txt")
print(dm.fold_1_x.shape)
print(dm.fold_1_y.shape)
print(dm.fold_2_x.shape)
print(dm.fold_2_y.shape)
network = MLP.MLP("test 5", 2, 7, 1)
for i in range(0, 5):
    print('started training{}'.format(i))
    network.backprop(dm.fold_1_x, dm.fold_1_y)
    network.test(dm.fold_2_x, dm.fold_2_y)
    dm.recombine()
network.print_results()