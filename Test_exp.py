import Experimenter
import MLP
import DataHandler

dh = DataHandler.DataHandler("news.csv", 1, token="")
network = MLP.MLP("news_test", dh.num_features,7, 1)
exp_1 = Experimenter.Experimenter(dh, network, 1)
exp_1.five_by_two()
