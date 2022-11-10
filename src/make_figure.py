import numpy as np
import seaborn as sns
import pandas as pd
from numpy.random import default_rng
import matplotlib.pyplot as plt
from gradient_descent import *
from learning_rates import *
from misc import *
from sklearn.model_selection import train_test_split

def guess_initial_theta(x,y,n_features):
    theta_init = np.zeros((n_features, 1))
    '''theta_init[0] = np.mean(y)
    theta_init[1] = (np.max(y)-np.min(y))/(np.max(x) - np.min(x))'''
    return theta_init

def make_dataframe_sgd(x, y, eta_method = 'basic', n_features = 3,n_iterations = 100,n_predictions = 1000):
    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    rng = default_rng()


    data = {'learning_rates' : {}, 'number_of_batches' : {}, 'lmbdas' : {}, 'mse' : {}}
    df = pd.DataFrame(data)
    theta_init = guess_initial_theta(X_train[1],y_train,n_features)

    for i in range(n_predictions):
        rate = rng.uniform(0.02,0.1)
        batch = rng.integers(1,20)
        lmbda = 10**rng.uniform(-8,-1)

        eta = make_adaptive_learner(eta_method,n_features,rate)
        theta0 = np.copy(theta_init)
        theta_final = gradient_descent(X_train, y_train, theta0, lmbda, n_iterations, eta, batch, 0)[-1]
        y_pred = X_test @ theta_final

        df.loc[len(df.index)] = [rate,batch,np.log10(lmbda),MSE(y_pred,y_test)]

    return df

def make_sgd_pairplot(eta_method, x, y, n_features, n_iterations, n_predictions):

    df = make_dataframe_sgd(x, y, eta_method, n_features = n_features,n_iterations = n_iterations, n_predictions = n_predictions)
    g = sns.PairGrid(df,hue = "mse")
    g.map_offdiag(sns.scatterplot)
    

    g.add_legend()
    g.fig.subplots_adjust(top=0.94)
    g.fig.suptitle('Method =' + eta_method)
    
    plt.show()



def sgd_figures(x, y, eta_method = 'basic', n_features = 3,n_iterations = 100):

    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    mse = np.zeros(n_iterations)
    iterations = np.arange(0,n_iterations,1)
    
    learning_rates = np.array([10**i for i in range(-5,0)])
    theta_init = guess_initial_theta(X_train[1],y_train,n_features)
    plt.subplot(221)
    for learning_rate in learning_rates:
        eta = make_adaptive_learner(eta_method,n_features,learning_rate)
        theta0 = np.copy(theta_init)
        thetas = gradient_descent(X_train, y_train, theta0, 0, n_iterations, eta, 1, 0)
        for i in iterations:

            y_pred = X_test @ thetas[i]
            mse[i] = MSE(y_pred,y_test)

        plt.plot(iterations,np.log(mse),label = 'Rate = %.0e' % (learning_rate))

    plt.title('Comparison of learning rates')
    plt.legend()


    plt.subplot(222)
    batches = np.arange(1,10,2)
    learning_rate = 0.1
    eta = make_adaptive_learner(eta_method,n_features,learning_rate)
    for n_batches in batches:
        theta0 = np.copy(theta_init)
        thetas = gradient_descent(X_train, y_train, theta0, 0, n_iterations, eta, n_batches, 0)
        for i in iterations:

            y_pred = X_test @ thetas[i]
            mse[i] = MSE(y_pred,y_test)

        plt.plot(iterations,np.log(mse),label = '%d batches' % (n_batches))

    plt.title('Comparison of numbers of batches')
    plt.legend()

    plt.subplot(223)
    lmbdas = np.array([0] + [10**i for i in range(-5,0)])
    learning_rate = 0.1
    eta = make_adaptive_learner(eta_method,n_features,learning_rate)
    for lmbda in lmbdas:
        theta0 = np.copy(theta_init)
        thetas = gradient_descent(X_train, y_train, theta0, lmbda, n_iterations, eta, 1, 0)
        for i in iterations:

            y_pred = X_test @ thetas[i]
            mse[i] = MSE(y_pred,y_test)

        plt.plot(iterations,np.log(mse),label = r'$\lambda$: %.0e' % (lmbda))

    plt.title('Comparison of lambdas')
    plt.legend()


    plt.subplot(224)
    batches = np.arange(1,10,2)
    learning_rate = 0.1
    lmbda = 0
    eta_algos = ['basic','ada','rms','adam']
    for algo in eta_algos:
        eta = make_adaptive_learner(algo,n_features,learning_rate)
        theta0 = np.copy(theta_init)
        thetas = gradient_descent(X_train, y_train, theta0, 0, n_iterations, eta, n_batches, 0)
        for i in iterations:

            y_pred = X_test @ thetas[i]
            mse[i] = MSE(y_pred,y_test)

        plt.plot(iterations,np.log(mse),label = algo)

    plt.title('Comparison of algorithms')
    plt.legend()
    plt.show()

