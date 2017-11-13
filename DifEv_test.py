import Experimenter
import MLP
import DataHandler

dh = DataHandler.DataHandler("Data/Music.txt", 1, token="")
network = MLP.MLP("dif_ev_music", dh.num_features,7, 1)
exp_1 = Experimenter.Experimenter(dh, network, 2)
exp_1.five_by_two()
