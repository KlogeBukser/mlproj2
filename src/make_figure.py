import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gradient_descent_iterators import *
from misc import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from numpy.random import default_rng

def save_figure(filename):
    file_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots"       # Finds the plot folder even when ran from outside src
    full_path = os.path.join(file_dir, filename)
    plt.savefig(full_path)


def get_sgd_iterator(X, y, theta_init, learning_rate = 0.001, lmbda = 0, n_batches = 1, momentum = 0,algo = 'SGD',logistic = False):

    if algo == 'ADA':
        return ADA(X,y,theta_init,learning_rate, lmbda, n_batches, momentum,logistic = logistic)
    if algo == 'RMS':
        return RMSProp(X,y,theta_init,learning_rate, lmbda, n_batches, momentum,logistic = logistic)
    if algo == 'ADAM':
        return ADAM(X,y,theta_init,learning_rate, lmbda, n_batches, momentum,logistic = logistic)

    return Gradient_descent(X,y,theta_init,learning_rate, lmbda, n_batches, momentum,logistic = logistic)


def rates_plot(X, y, learning_rates = np.linspace(-7,-5), algo = 'SDG',n_batches = 2,logistic = False):
    """ Makes momentum plot for standard gradient descent algorithm. """

    n_features = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    filename = "learning_rate" + algo + "_" + str(n_batches) + ".png"
    if logistic:
        score_func = accuracy_score
        score_name="Accuracy score"
        
        filename = "logi_" + filename
        plt.title(score_name + " for initial learning rates with " + algo + " regression  \n learning rates $\in [10^{%d},10^{%.2f}])$ | minibatches: %d" % (learning_rates[0],learning_rates[-1],n_batches))
        plt.xscale("log")
        learning_rates = 10**learning_rates

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    else:
        score_func = MSE
        score_name="MSE"
        plt.title(score_name + " for initial learning rates with " + algo + " regression  \n learning rates $\in [%.2f,%.2f]$ | minibatches: %d" % (learning_rates[0],learning_rates[-1],n_batches))
        plt.yscale("log")

    
    

    theta_init = np.zeros((n_features, 1))

    for n_epochs in [20,50,100,200]:
        
        score = np.empty(len(learning_rates))
        for i in range(len(learning_rates)):
            iterator = get_sgd_iterator(X_train,y_train,theta_init,learning_rate=learning_rates[i],n_batches=n_batches,algo=algo,logistic=logistic)

            iterator.advance(n_epochs)
            y_pred = iterator.predict(X_test)
            score[i] = score_func(y_pred,y_test)
        plt.plot(learning_rates,score,label = " %d epochs" % (n_epochs))

    plt.xlabel("Learning rate")
    plt.ylabel(score_name)
    plt.legend()
    save_figure(filename)
    plt.close()




def comparison_plots(X,y,learning_rate_range = np.linspace(0.01,0.15,100), lmbda_range = [-8,-1],algos = ['SGD','ADA','RMS','ADAM'],filename = "comparisons.pdf",logistic = False):
    rng = default_rng(1)

    n_predictions = 200
    n_epochs = 100
    momentum = 0
    n_batches = 1

    X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    theta_init = np.zeros((X_train.shape[1],1))

    learning_rates = np.linspace(*learning_rate_range)
    if logistic:
        score_func = accuracy_score
        score_name="Accuracy score"
        learning_rates = 10**learning_rates
        rate_func = log10_default_func
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


    else:
        score_func = logMSE
        score_name = "MSE"
        rate_func = default_func

    data = {'learning_rates' : {}, '$\lambda$ (log$_{10}$)' : {}, score_name : {}, 'Algorithm' : {}}
    df = pd.DataFrame(data)

    for i in range(n_predictions):
        rate = rng.choice(learning_rates) 
        lmbda = 10**(rng.uniform(*lmbda_range))
        algo = rng.choice(algos)
        iterator = get_sgd_iterator(X_train,y_train, theta_init, rate, lmbda, n_batches, momentum, algo = algo,logistic=logistic)
        
        iterator.advance(n_epochs)
        y_pred = iterator.predict(X_test)
        score = score_func(y_pred,y_test)
        df.loc[len(df.index)] = [rate_func(rate),np.log10(lmbda),score,algo]

    

    g = sns.pairplot(data=df, x_vars=['learning_rates', '$\lambda$ (log$_{10}$)'], y_vars = [score_name], hue='Algorithm')
    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle(score_name + ' after ' + str(n_epochs) + ' iterations \n (momentum: %.2f | minibatches: %d )' % (momentum,n_batches))
    save_figure(filename)
    plt.close()



def convergence_plots(X,y,algo = 'SDG', learning_rate = 0.3, batch_list = [1,2,5,8],bootstraps = 10):
    
    rng = default_rng(1)

    n_predictions = 200
    lmbda = 0
    n_epochs = 100
    stop_crit = 1e-2
    min_epochs = 5

    X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    n_training_points = len(y_train)
    theta_init = np.zeros((X_train.shape[1],1))

    momentum_range = [0,1]
    
    data = {'epochs' : {}, 'MSE (log)' : {}, 'Minibatches' : {}, 'momentum' : {}}
    df = pd.DataFrame(data)


    for i in range(n_predictions):
        n_batches = rng.choice(batch_list)
        momentum = rng.uniform(*momentum_range)
        epoch = 0
        mse  = 0
        for j in range(bootstraps):
            indices = rng.integers(0,n_training_points,n_training_points)
            iterator = get_sgd_iterator(X_train[indices],y_train[indices],theta_init,learning_rate=learning_rate,lmbda = lmbda,n_batches = n_batches,momentum=momentum,algo=algo)
            epoch += iterator.advance(n_epochs = n_epochs, stop_crit = stop_crit, min_epochs = min_epochs)
            mse += MSE(iterator.predict(X_test),y_test)

        df.loc[len(df.index)] = [epoch/bootstraps, np.log(mse/bootstraps), n_batches, momentum]

    df["Minibatches"] = df["Minibatches"].astype(int)

    g = sns.relplot(data=df, x='momentum', y ='epochs',hue = 'Minibatches', palette='bright',markers='.')
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Convergence of ' + algo + ' regression model \n with momentum and number of minibatches as parameters \n (bootstraps: %d | learning rate: %.2f | $\lambda$: %.2f )' % (bootstraps,learning_rate,lmbda))

    save_figure(algo + 'convergence.pdf')
    plt.close()

    for momentum in [0,0.2,0.4,0.6,0.8]:
        for n_batches in batch_list:
            epochs = np.arange(n_epochs)
            mse = np.zeros(n_epochs)
            iterator = get_sgd_iterator(X_train,y_train,theta_init,learning_rate=learning_rate,lmbda = lmbda,n_batches = n_batches,momentum=0,algo=algo)
            for epoch in range(n_epochs):
                iterator.advance()
                pred = iterator.predict(X_test)
                mse[epoch] = np.log(MSE(pred,y_test))
            
            plt.plot(epochs,mse,label = '%d minibatches' % (n_batches))

        plt.title("MSE over epochs with " + algo + " regression \n (momentum: %.2f | learning rate: %.2f | $\lambda$: %.2f ) " % (momentum,learning_rate,lmbda))
        plt.xlabel("Number of epochs")
        plt.ylabel("MSE (log)")
        plt.legend()
        save_figure(algo + "_" + str(momentum) +  "_mse.pdf")
        plt.close()