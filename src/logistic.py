import numpy as np
from sklearn.datasets import load_breast_cancer
from make_figure import rates_plot,comparison_plots



df = load_breast_cancer()

X = df.data
y = df.target[:,np.newaxis]



rates_plot(X, y, learning_rates = np.linspace(-8,0,200), algo = 'SGD',n_batches=1,logistic=True)
rates_plot(X, y, learning_rates = np.linspace(-8,0,200), algo = 'RMS',n_batches=1,logistic=True)
rates_plot(X, y, learning_rates = np.linspace(-8,0,200), algo = 'ADA',n_batches=1,logistic=True)
rates_plot(X, y, learning_rates = np.linspace(-8,0,200), algo = 'ADAM',n_batches=1,logistic=True)

comparison_plots(X,y,learning_rate_range = [-7,-5],lmbda_range = [-6,4],algos = ['SGD'],filename = "logi_comparison.png",logistic=True)
