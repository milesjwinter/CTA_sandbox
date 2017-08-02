import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.misc import factorial
import scipy.stats as stats

def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)


def negLogLikelihood(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl


# get poisson deviated random numbers
data = np.random.poisson(2, 1000)

# minimize the negative log-Likelihood

result = minimize(negLogLikelihood,  # function to minimize
                  x0=np.ones(1),     # start value
                  args=(data,),      # additional arguments for function
                  method='Powell',   # minimization method, see docs
                  )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
print(result)

# plot poisson-deviation with fitted parameter
x_plot = np.linspace(0, 20, 1000)
hist_data = np.histogram(data,bins=np.linspace(0,20,1000))

print stats.chisquare(f_obs=data,f_exp=poisson(x_plot, result.x))

plt.hist(data, bins=np.arange(15) - 0.5, normed=True)
plt.plot(x_plot, poisson(x_plot, result.x), 'r-', lw=2)
plt.show()
