import numpy as np
from numpy.random import default_rng
from make_figure import *

rng = default_rng(seed = 5473)


# Makes folder for holding plots if it doesn't already exist
plot_path = os.path.dirname(os.path.abspath(__file__)) + "/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

