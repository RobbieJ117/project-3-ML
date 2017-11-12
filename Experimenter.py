import MLP
import numpy as np
import numba
import sklearn as sk
class Experimenter:
    def __init__(self, dm, mlp, training_method, beta=.5, pr=.1, maxgen=120):
        self.dm = dm
        self.mlp = mlp
        self.train_method = training_method
        self.populatoin = []
        self.beta = beta
        self.pr = pr
        self.maxgen=maxgen

    #Calls different training methods within
    def train(self, features, targets):
        if (self.train_method==1):
            self.mlp.backprop(features, targets)
        elif(self.train_method==2):
            self.mlp.difEvoTrain(self.beta, self.pr)
        elif(self.train_method==3):
            #Initialize population once on first training call
            if (self.mlp.iteration==0):
                self.mlp.init_ES(len(features), self.maxgen)
            self.mlp.train_ES(features, targets)


    def test(self, features, targets):
        self.mlp.test(features, targets)

    def five_by_two(self):
        for i in range(0, 5):
            self.dm.recombine(2)
            x1, y1 = self.dm.cleave(self.dm.folds_container[0])
            self.train(x1, y1)
            x2, y2 = self.dm.cleave(self.dm.folds_container[1])
            self.test(x2, y2)
            print("\nIteration {} complete".format(i+1))
        self.mlp.print_results()