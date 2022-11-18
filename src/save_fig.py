import os
import matplotlib.pyplot as plt

def save_figure(filename):
    # Makes folder for holding plots if it doesn't already exist
    file_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    full_path = os.path.join(file_dir, filename)
    plt.savefig(full_path)