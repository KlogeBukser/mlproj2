import numpy as np
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import *

# the number of datapoints
n_datapoints = 100
n_iterations = 1000000
step_size = 0.1
momentum = 0.3
cond = 1e-4
batch_size = 5
x, y = gen_simple(n_datapoints)


ypredict = simple_descent(x,y, cond,n_iterations,step_size)
ypredict_mom = momentum_descent(x, y, cond, n_iterations, step_size, momentum)
ypredict_sdg = sgd(x, y, cond, n_iterations, step_size, batch_size)
ypredict_sdg_mom = sgd_mom(x, y, cond, n_iterations, step_size, batch_size, momentum)

plt.subplot(221)
plt.title(r'Stochastic gradient descent')
plt.plot(x, ypredict_sdg, '-b')
plt.plot(x, y ,'r.')
plt.ylabel('Without momentum')


plt.subplot(222)
plt.title(r'Gradient descent')
plt.plot(x, ypredict_mom, '-b')
plt.plot(x, y ,'r.')

plt.subplot(223)
plt.plot(x, ypredict_sdg_mom, '-b')
plt.plot(x, y ,'r.')
plt.ylabel('With momentum')

plt.subplot(224)
plt.plot(x, ypredict, '-b')
plt.plot(x, y ,'r.')


'''
ypredict_lin = lin_reg(x,y)
plt.plot(x, ypredict_lin, '-b')
plt.plot(x, y ,'r.')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
'''
plt.show()