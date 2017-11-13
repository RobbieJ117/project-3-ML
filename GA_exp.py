import Experimenter
import MLP
import DataHandler

dh = DataHandler.DataHandler("Data/News.csv", 1, token=",")
network = MLP.MLP("GA_News", dh.num_features,7, 1)
exp_1 = Experimenter.Experimenter(dh, network, 4, maxgen=150)
exp_1.five_by_two()