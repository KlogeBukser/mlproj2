import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gradient_descent_iterators import *
from misc import MSE, make_design_1D
from sklearn.model_selection import train_test_split

def save_figure(filename):
    file_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots"       # Finds the plot folder even when ran from outside src
    full_path = os.path.join(file_dir, filename)
    plt.savefig(full_path)


def get_sgd_iterator(X, y, theta_init, learning_rate = 0.001, lmbda = 0, n_batches = 1, momentum = 0,algo = 'basic'):

    if algo == 'ada':
        return ADA(X,y,theta_init,learning_rate, lmbda, n_batches, momentum)
    if algo == 'rms':
        return RMSProp(X,y,theta_init,learning_rate, lmbda, n_batches, momentum)
    if algo == 'adam':
        return ADAM(X,y,theta_init,learning_rate, lmbda, n_batches, momentum)

    return Gradient_descent(X,y,theta_init,learning_rate, lmbda, n_batches, momentum)


def momentum_plot(x,y,learning_rate,n_epochs,n_features,lmbda,n_batches = 1,momentums = np.arange(0,1,0.2),algo = 'basic',smooth = False):
    """ Makes momentum plot for standard gradient descent algorithm. """

    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    # Arrays for plotting
    epochs = np.arange(0,n_epochs,1)
    mses = np.empty(n_epochs)

    for momentum in momentums:
        theta_init = np.ones((n_features, 1))
        gd_iterator = get_sgd_iterator(X_train,y_train,theta_init,learning_rate,lmbda,n_batches,momentum,algo)
        for epoch in epochs:
            gd_iterator.advance()
            y_pred = gd_iterator.predict(X_test,smooth)
            mses[epoch] = MSE(y_pred,y_test)
            
        plt.plot(epochs,mses,label = "Momentum: %.2f" % (momentum))

    plt.title(" Momentum ")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    save_figure('momentum.png')
    plt.close()

def batches_plot(x,y,learning_rate,n_epochs,n_features,lmbda,n_batches_list = np.arange(1,20,3),momentum = 0, algo = 'basic', smooth = False):
    """ Makes momentum plot for standard gradient descent algorithm. """

    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    # Arrays for plotting
    epochs = np.arange(0,n_epochs,1)
    mses = np.empty(n_epochs)

    for n_batches in n_batches_list:
        theta_init = np.ones((n_features, 1))
        gd_iterator = get_sgd_iterator(X_train,y_train,theta_init,learning_rate,lmbda,n_batches,momentum,algo)
        for epoch in epochs:
            gd_iterator.advance()
            y_pred = gd_iterator.predict(X_test,smooth)
            mses[epoch] = MSE(y_pred,y_test)
            
        plt.plot(epochs,mses,label = "Number of batches: %d" % (n_batches))

    plt.title(" Number of minibatches ")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    save_figure('batches.png')
    plt.close()




def make_dataframe_sgd(x, y, params, n_features = 3,n_epochs = 100,n_predictions = 200):

    X = make_design_1D(x,n_features)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    rng = default_rng()


    data = {'learning_rates' : {}, 'number_of_batches' : {}, 'lmbdas' : {}, 'mse' : {}, 'algorithm' : {}}
    df = pd.DataFrame(data)
    theta_init = np.ones((n_features, 1))
    

    for i in range(n_predictions):
        rate, batch, lmbda, algo = params(rng)

        # X_train,y_train,theta_init,learning_rate,lmbda,n_batches,momentum,algo
        iterator = get_sgd_iterator(X_train,y_train, theta_init, rate, lmbda, batch,momentum = 0, algo = algo)
        iterator.advance(n_epochs)
        y_pred = iterator.predict(X_test)
        df.loc[len(df.index)] = [rate,batch,np.log10(lmbda),MSE(y_pred,y_test),algo]

    return df


def make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions,filename):

    df = make_dataframe_sgd(x, y, params, n_features,n_iterations, n_predictions)
    g = sns.pairplot(data=df, x_vars=['learning_rates', 'number_of_batches', 'lmbdas'], y_vars = ['mse'], hue='algorithm')
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean squared error after ' + str(n_iterations) + ' iterations')
    save_figure(filename)
    plt.close()