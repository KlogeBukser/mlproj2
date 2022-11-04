import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import *
from learning_rates import *
from misc import make_design_1D

def make_subplot(eta_method = 'basic', n_features = 3,learning_rate = 0.5,batch_size = 5):
    n_datapoints = 100
    n_iterations = 10000
    momentum = 0.3
    cond = 1e-4
    batch_size = 5
    x, y = gen_simple(n_features, n_datapoints)
    X = make_design_1D(x,n_features)
    theta0 = np.ones((n_features,1))*0.5

    print(eta_method)

    if eta_method == 'basic':
        eta = Basic_learning_rate(learning_rate)
    if eta_method == 'ada':
        eta = ADA(n_features,learning_rate)


    x_new = np.linspace(np.min(x),np.max(x),n_datapoints)
    X_new = make_design_1D(x_new,n_features)

    plt.subplot(221)
    plt.title(r'Stochastic gradient descent')
    ypredict_sdg = X_new @ sgd(X, y, np.copy(theta0), cond, n_iterations, eta, batch_size)
    plt.plot(x_new, ypredict_sdg, '-b')
    plt.plot(x, y ,'r.')
    plt.ylabel('Without momentum')
    eta.reset()


    plt.subplot(222)
    plt.title(r'Gradient descent')
    ypredict_mom = X_new @ momentum_descent(X, y, np.copy(theta0), cond, n_iterations, eta, momentum)
    plt.plot(x_new, ypredict_mom, '-b')
    plt.plot(x, y ,'r.')
    eta.reset()

    plt.subplot(223)
    ypredict_sdg_mom = X_new @ sgd_mom(X, y, np.copy(theta0), cond, n_iterations, eta, batch_size, momentum)
    plt.plot(x_new, ypredict_sdg_mom, '-b')
    plt.plot(x, y ,'r.')
    plt.ylabel('With momentum')
    eta.reset()

    plt.subplot(224)
    ypredict = X_new @ simple_descent(X, y, np.copy(theta0), cond, n_iterations, eta)
    plt.plot(x_new, ypredict, '-b')
    plt.plot(x, y ,'r.')

    plt.show()