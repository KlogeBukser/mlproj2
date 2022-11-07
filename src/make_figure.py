import numpy as np
import seaborn as sns
from numpy.random import default_rng
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import *
from learning_rates import *
from misc import *


def sgd_figures(x, y, eta_method = 'basic', n_features = 3,n_iterations = 100):

    X = make_design_1D(x,n_features)

    mse = np.zeros(n_iterations)
    iterations = np.arange(0,n_iterations,1)
    
    learning_rates = np.array([10**i for i in range(-5,0)])

    plt.subplot(221)
    for learning_rate in learning_rates:
        eta = make_adaptive_learner(eta_method,n_features,learning_rate)
        theta0 = np.ones((n_features,1))
        thetas = gradient_descent(X, y, theta0, 0, n_iterations, eta, 1, 0)
        for i in iterations:

            y_pred = X @ thetas[i]
            mse[i] = MSE(y_pred,y)

        plt.plot(iterations,np.log(mse),label = 'Rate = %.0e' % (learning_rate))

    plt.title('Comparison of learning rates')
    plt.legend()


    plt.subplot(222)
    batches = np.arange(1,10,2)
    learning_rate = 0.1
    eta = make_adaptive_learner(eta_method,n_features,learning_rate)
    for n_batches in batches:
        theta0 = np.ones((n_features,1))
        thetas = gradient_descent(X, y, theta0, 0, n_iterations, eta, n_batches, 0)
        for i in iterations:

            y_pred = X @ thetas[i]
            mse[i] = MSE(y_pred,y)

        plt.plot(iterations,np.log(mse),label = '%d batches' % (n_batches))

    plt.title('Comparison of numbers of batches')
    plt.legend()

    plt.subplot(223)
    lmbdas = np.array([0] + [10**i for i in range(-5,0)])
    learning_rate = 0.1
    eta = make_adaptive_learner(eta_method,n_features,learning_rate)
    for lmbda in lmbdas:
        theta0 = np.ones((n_features,1))
        thetas = gradient_descent(X, y, theta0, lmbda, n_iterations, eta, 1, 0)
        for i in iterations:

            y_pred = X @ thetas[i]
            mse[i] = MSE(y_pred,y)

        plt.plot(iterations,np.log(mse),label = 'lambda: %.0e' % (lmbda))

    plt.title('Comparison of lambdas')
    plt.legend()


    plt.subplot(224)
    batches = np.arange(1,10,2)
    learning_rate = 0.1
    lmbda = 0
    eta_algos = ['basic','ada','rms','adam']
    for algo in eta_algos:
        eta = make_adaptive_learner(algo,n_features,learning_rate)
        theta0 = np.ones((n_features,1))
        thetas = gradient_descent(X, y, theta0, 0, n_iterations, eta, n_batches, 0)
        for i in iterations:

            y_pred = X @ thetas[i]
            mse[i] = MSE(y_pred,y)

        plt.plot(iterations,np.log(mse),label = algo)

    plt.title('Comparison of algorithms')
    plt.legend()
    plt.show()


def make_subplot(eta_method = 'basic', n_features = 3,learning_rate = 0.01,batch_size = 5):
    n_datapoints = 100
    n_iterations = 100
    momentum = 0.3
    batch_size = 5
    x, y = gen_simple(n_features, n_datapoints)
    X = make_design_1D(x,n_features)
    theta0 = np.ones((n_features,1))*0.5

    eta = make_adaptive_learner(eta_method,n_features,learning_rate)


    x_new = np.linspace(np.min(x),np.max(x),n_datapoints)
    X_new = make_design_1D(x_new,n_features)

    plt.subplot(221)
    plt.title(r'Stochastic gradient descent')
    ypredict_sdg = X_new @ sgd(X, y, np.copy(theta0), n_iterations, eta, batch_size)
    plt.plot(x_new, ypredict_sdg, '-b')
    plt.plot(x, y ,'r.')
    plt.ylabel('Without momentum')
    eta.reset()


    plt.subplot(222)
    plt.title(r'Gradient descent')
    ypredict_mom = X_new @ momentum_descent(X, y, np.copy(theta0), n_iterations, eta, momentum)
    plt.plot(x_new, ypredict_mom, '-b')
    plt.plot(x, y ,'r.')
    eta.reset()

    plt.subplot(223)
    ypredict_sdg_mom = X_new @ gradient_descent(X, y, np.copy(theta0), n_iterations, eta, batch_size, momentum)
    plt.plot(x_new, ypredict_sdg_mom, '-b')
    plt.plot(x, y ,'r.')
    plt.ylabel('With momentum')
    eta.reset()

    plt.subplot(224)
    ypredict = X_new @ simple_descent(X, y, np.copy(theta0), n_iterations, eta)
    plt.plot(x_new, ypredict, '-b')
    plt.plot(x, y ,'r.')

    plt.show()