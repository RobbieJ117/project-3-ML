import Experimenter
import MLP
from Data import DataHandler

dh = DataHandler.DataHandler("3d_test.txt", 1, token=",")
network = MLP.MLP("dif_ev_3dTest", dh.num_features,7, 1)
exp_1 = Experimenter.Experimenter(dh, network, 2)
exp_1.five_by_two()
