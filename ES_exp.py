import Experimenter
import MLP
import DataHandler

dh = DataHandler.DataHandler("Data/MusicPlus.txt", 1, token=",")
network = MLP.MLP("ES_MusicPlus", dh.num_features,7, 1)
exp_1 = Experimenter.Experimenter(dh, network, 3, maxgen=150)
exp_1.five_by_two()