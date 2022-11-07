import numpy as np
import seaborn as sns
from numpy.random import default_rng
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import *
from learning_rates import *
from misc import *
from make_figure import make_subplot


n_datapoints = 100
n_iterations = 100
n_features = 3
n_learning_rates = 100
M = 5
momentum = 0.3
batch_size = 5
x, y = gen_simple(n_features, n_datapoints)
X = make_design_1D(x,n_features)
rng = default_rng()
theta0 = np.ones((n_features,1))*0.5
learning_rate = 0.01




eta_method = 'basic'

eta = make_adaptive_learner(eta_method,n_features,learning_rate)

'''simple_thetas = simple_descent(X, y, theta0, cond,n_iterations, eta)
eta.reset()

momentum_thetas = momentum_descent(X, y, theta0, cond,n_iterations, eta, momentum)
eta.reset()

sgd_thetas = sgd(X, y, theta0, cond,n_iterations, eta, M)
eta.reset()

sgd_mom_thetas = gradient_descent(X, y, theta0, cond,n_iterations, eta, M, momentum)
eta.reset()'''




x_new = np.linspace(np.min(x),np.max(x),n_datapoints)
X_new = make_design_1D(x_new,n_features)
mse = np.zeros(n_datapoints)
momentums = np.arange(0,1,0.2)
iterations = np.arange(0,n_datapoints,1)

for momentum in momentums:
    thetas = momentum_descent(X, y, theta0, n_iterations, eta, momentum)
    eta.reset()
    for i in iterations:

        y_pred = X_new @ thetas[i]
        mse[i] = MSE(y_pred,y)

    plt.plot(iterations,mse,label = 'momentum = %.2f' % (momentum))

plt.title('Momentum plot')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.legend()
plt.show()

#make_subplot(eta_method = 'adam', n_features = 3,learning_rate = 0.01, batch_size = 5)


