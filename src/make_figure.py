import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gradient_descent_iterators import *
from misc import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from numpy.random import default_rng



def save_figure(filename):
    # Makes folder for holding plots if it doesn't already exist
    file_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
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

    filename = "learning_rate" + algo + "_" + str(n_batches) + ".pdf"
    if logistic:
        score_func = accuracy_score
        score_name="Accuracy score"
        
        filename = "logi_" + filename
        plt.title(score_name + " for initial learning rates with " + algo + " regression  \n learning rates $\in [10^{%d},10^{%d}])$ | minibatches: %d" % (learning_rates[0],learning_rates[-1],n_batches))
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




def comparison_plots(X,y,learning_rate_range = np.linspace(0.01,0.15,100), lmbda_range = [-8,-1],algos = ['SGD','ADA','RMS','ADAM'],n_batches = 1, filename = "comparisons.pdf",logistic = False):
    rng = default_rng(1)

    n_predictions = 200
    n_epochs = 100
    momentum = 0

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



def convergence_plots(X,y,algo = 'SDG', learning_rate = 0.3, batch_list = [1,2,5,8],bootstraps = 10,stop_crit = 1e-2):
    
    rng = default_rng(1)

    n_predictions = 200
    lmbda = 0
    n_epochs = 100
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
    g.fig.suptitle('Convergence of ' + algo + ' regression model \n with momentum and number of minibatches as parameters \n (bootstraps: %d | learning rate: %.2f | $\lambda$: %.2f | tolerance: %.2f )' % (bootstraps,learning_rate,lmbda,stop_crit))

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


def hyper_matrix(X,y,min_rate = -8, max_rate = 4, min_lmb = -8, max_lmb = 2, n_batches = 20, n_epochs = 10, algo = "SGD",logistic = False,rate_points = 1):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    if logistic:
        score_func = accuracy_score
        title = "Accuracy score for logistic regression with " + algo + " method"
        filename = algo + "_logistic_clf.pdf"

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        learning_rates = np.logspace(min_rate, max_rate, rate_points*(max_rate-min_rate) + 1)
        

    else:
        score_func = logMSE
        title = "MSE score for " + algo + " regression"
        filename = algo + "_logistic_reg.pdf"

        #learning_rates = np.round(np.linspace(min_rate*1e-2, max_rate*1e-2, max_rate-min_rate+1),2)


    learning_rates = np.logspace(min_rate, max_rate, rate_points*(max_rate-min_rate) + 1)
    lmbd_vals = np.logspace(min_lmb, max_lmb, max_lmb-min_lmb+1)


    data = {'learning rate' : {}, 'lambda' : {}, 'Score' : {}}
    df = pd.DataFrame(data)
    

    theta = np.zeros((X_train.shape[1],1))

    for learning_rate in learning_rates:
        for lmbd in lmbd_vals:
            solver = get_sgd_iterator(X_train,y_train,theta,learning_rate,lmbd,n_batches, algo = algo, logistic = logistic)
            solver.advance(n_epochs)
            y_pred = solver.predict(X_test)
            score = score_func(y_pred,y_test)
            df.loc[len(df.index)] = [np.log10(learning_rate),np.log10(lmbd),score]

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(df.pivot("learning rate", "lambda", "Score"), annot=True, ax=ax, cmap="viridis")
    ax.set_title(title)
    ax.set_ylabel("Learning rate: log$_{10}(\eta)$")
    ax.set_xlabel("Regularization parameter: log$_{10}(\lambda$)")
    save_figure(filename)
    plt.close()



def log_reg_sklearn(X, y, learning_rate, lmbda, n_epochs=100, predictions = 10):

    data = {'Model' : {}, 'Accuracy score' : {}}
    df = pd.DataFrame(data)
    

    for i in range(predictions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

        # Make a copy for the sklear model, to make sure the data isn't changed before its used for the other models
        sk_scaler = StandardScaler()
        lr = LogisticRegression()
        X_train_sk = np.copy(X_train)
        X_test_sk = np.copy(X_test)

        y_train_sk = np.copy(y_train).ravel()
        y_test_sk = np.copy(y_test).ravel()

        model = Pipeline([('standardize', sk_scaler),('log_reg', lr)])
        model.fit(X_train_sk, y_train_sk)
        sk_pred = model.predict(X_test_sk)
        sk_score = accuracy_score(sk_pred,y_test_sk)
        df.loc[len(df.index)] = ["sklearn",sk_score]

        # Our models Models
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        theta = np.zeros((X_train.shape[1],1))

        for algo in ["SGD","ADA","RMS","ADAM"]:
            our_model = get_sgd_iterator(X_train,y_train,theta,learning_rate=learning_rate,lmbda=lmbda,algo = algo, logistic = True)
            our_model.advance(n_epochs)
            pred = our_model.predict(X_test)
            score = accuracy_score(pred,y_test)
            df.loc[len(df.index)] = [algo, score]

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.boxplot(data=df, x="Model", y="Accuracy score",ax = ax)
    ax.set_title("Accuracy scores with logistic regression \n (Data samplings : %d | number of epochs : %d | Learning rate : %.2g)" % (predictions,n_epochs,learning_rate))
    save_figure("Sklearn_comp.pdf")
    plt.close()