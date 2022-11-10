import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from gradient_descent import *
from learning_rates import *
from misc import *


def guess_initial_theta(x,y,n_features):
    theta_init = np.ones((n_features, 1))
    '''theta_init[0] = np.mean(y)
    theta_init[1] = (np.max(y)-np.min(y))/(np.max(x) - np.min(x))'''
    return theta_init

def make_dataframe_sgd(x, y, n_features = 3,n_iterations = 100,n_predictions = 1000, eta_algos = ['basic','ada','rms','adam']):
    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    rng = default_rng()


    data = {'learning_rates' : {}, 'number_of_batches' : {}, 'lmbdas' : {}, 'mse' : {}, 'eta' : {}}
    df = pd.DataFrame(data)
    theta_init = guess_initial_theta(X_train[1],y_train,n_features)

    for i in range(n_predictions):
        rate = rng.uniform(0.02,0.15)
        batch = rng.integers(1,20)
        lmbda = 10**rng.uniform(-8,-1)
        eta_method = rng.choice(eta_algos)

        eta = make_adaptive_learner(eta_method,n_features,rate)
        theta0 = np.copy(theta_init)
        theta_final = np.mean(gradient_descent(X_train, y_train, theta0, lmbda, n_iterations, eta, batch, 0)[-2:],axis = 0)
        y_pred = X_test @ theta_final

        df.loc[len(df.index)] = [rate,batch,np.log10(lmbda),np.log(MSE(y_pred,y_test)),eta_method]

    return df


def make_epoch_plot(x, y, n_iterations,n_features,lmbda,eta_method,n_batches,learning_rate):

    iterations = np.arange(0,n_iterations,1)
    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    eta = make_adaptive_learner(eta_method,n_features,learning_rate)

    
    theta_init = guess_initial_theta(X_train[1],y_train,n_features)
    thetas = gradient_descent(X_train, y_train, theta_init, lmbda, n_iterations, eta, n_batches, 0)
    mses = np.empty(thetas.shape[0])
    for i in iterations:
        y_pred = X_test @ thetas[i]
        mses[i] = MSE(y_test,y_pred)

    plt.title("MSE over iterations with " + eta_method + " algorithm. (" + str(n_batches) + " minibatches)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.plot(iterations,mses)
    plt.show()



def make_sgd_compare_plot(x, y, n_features, n_iterations, n_predictions):

    df = make_dataframe_sgd(x, y, n_features = n_features,n_iterations = n_iterations, n_predictions = n_predictions, eta_algos = ['basic','ada','rms','adam'])
    
    
    g = sns.pairplot(data=df, x_vars=['learning_rates', 'number_of_batches', 'lmbdas'], y_vars = ['mse'], hue='eta')
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean squared error after ' + str(n_iterations) + ' iterations')
    plt.show()

    df = df.drop(df[df['eta'] == 'ada'].index)
    g = sns.pairplot(data=df, x_vars=['learning_rates', 'number_of_batches', 'lmbdas'], y_vars = ['mse'], hue='eta')
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean squared error after ' + str(n_iterations) + ' iterations')
    plt.show()
