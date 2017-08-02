import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.misc import factorial


#Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2.*sigma**2))

def negLogLikelihood(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(gauss_function(data, params[0],params[1],params[2])))
    return lnl


# get poisson deviated random numbers
charge = np.loadtxt('total_charge.txt')

# minimize the negative log-Likelihood

result = minimize(negLogLikelihood,  # function to minimize
                  x0=np.ones(1),     # start value
                  args=(charge,),      # additional arguments for function
                  method='Powell',   # minimization method, see docs
                  )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
print(result)

# plot poisson-deviation with fitted parameter
x_plot = np.linspace(0, 20, 1000)

plt.hist(data, bins=np.arange(15) - 0.5, normed=True)
plt.plot(x_plot, gauss_function(x_plot, result.x), 'r-', lw=2)
plt.show()
