import numpy as np
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import simple_descent, lin_reg, momentum_descent

# the number of datapoints
n_datapoints = 100
n_iterations = 10000
step_size = 0.1
momentum = 0.3
cond = 1e-10
x, y = gen_simple(n_datapoints)

ypredict_lin = lin_reg(x,y)
ypredict = simple_descent(x,y, cond,n_iterations,step_size)
ypredict_mom = momentum_descent(x, y, cond, n_iterations, step_size, momentum)

plt.subplot(311)
plt.title(r'Gradient descent')
plt.plot(x, ypredict_mom, "b-", label = 'Gradient descent with momentum')
plt.plot(x, y ,'r.')
plt.ylabel(r'$y$')
plt.legend()

plt.subplot(312)
plt.plot(x, ypredict, "b-", label = 'Gradient descent')
plt.plot(x, y ,'r.')
plt.ylabel(r'$y$')
plt.legend()

plt.subplot(313)
plt.plot(x, ypredict_lin, "b-", label = 'Linear regression')
plt.plot(x, y ,'r.')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()

plt.show()