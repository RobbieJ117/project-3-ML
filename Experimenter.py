import MLP
import numpy as np
import numba
import sklearn as sk
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