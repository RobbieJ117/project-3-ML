import MLP
import Experimenter
import DataHandler

dh = DataHandler.DataHandler("news.csv", 1, delimiter="")
network = MLP.MLP("news_test", dm.num_features,7, 1)
exp_1 = Experimenter.Experimenter(dm, network, 1)
exp_1.five_by_two()
