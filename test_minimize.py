import numpy as np
import pylab as pl
from scipy.optimize import minimize

points = 500
xlim = 3.

def f(x,*p):
    a1,a2,a3,a4,a5 = p
    return a1*np.abs(x-a2)**a3 * np.exp(-a4 * np.abs(x)**a5)

# generate noisy data with known coefficients
p0 = [1.4,-.8,1.1,1.2,2.2]
x = (np.random.rand(points) * 2. - 1.) * xlim
x.sort()
y = f(x,*p0)
y_noise = y + np.random.randn(points) * .05

# mean squared error wrt. noisy data as a function of the parameters
err = lambda p: np.mean((f(x,*p)-y_noise)**2)

# bounded optimization using scipy.minimize
p_init = [1.,-1.,.5,.5,2.]
p_opt = minimize(
    f, # minimize wrt to the noisy data
    p_init, 
    bounds=[(None,None),(-1,1),(None,None),(0,None),(None,None)], # set the bounds
    method="L-BFGS-B" # this method supports bounds
).x

# plot everything
pl.scatter(x, y_noise, alpha=.2, label="f + noise")
pl.plot(x, y, c='#000000', lw=2., label="f")
pl.plot(x, f(x,*p_opt) ,'--', c='r', lw=2., label="fitted f")

pl.xlabel("x")
pl.ylabel("f(x)")
pl.legend(loc="best")
pl.xlim([-xlim*1.01,xlim*1.01])

pl.show()
