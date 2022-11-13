from generate import gen_simple
import numpy as np
from numpy.random import default_rng
from make_figure import *
from hyperparams import Hyperparams
from logistic import logistic_reg

rng = default_rng(seed = 5473)



plot_path = os.path.dirname(os.path.abspath(__file__)) + "/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


simple_plots()
comparison_plots()
logistic_reg()