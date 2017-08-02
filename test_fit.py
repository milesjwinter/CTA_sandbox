import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def GaussSum(x,*p):
    n=len(p)/3
    A=p[:n]
    w=p[n:2*n]
    c=p[2*n:3*n]
    y = sum([ A[i]*np.exp(-(x-c[i])**2./(2.*(w[i])**2.))/(2*np.pi*w[i]**2)**0.5 for i in range(n)])
    return y

def SiPM_Fit(x,lam,mu,sig0,sig1,g,x0):
    f = sum([sum([(lam**p*np.exp(-lam)/np.math.factorial(p))*((p*mu)**s*np.exp(-p*mu)/np.math.factorial(s))*(np.exp(-((x-x0)/g-(p+s)**2)/(2*(sig0**2+(p+s)*sig1**2)))/(np.sqrt(2.*np.pi)*np.sqrt(sig0**2+(p+s)*sig1**2))) for p in range(5)]) for s in range(5)])
    #f = sum([sum([(lam**p*np.exp(-lam)/np.math.factorial(p))*((p**2*mu)**s*np.exp(-p**2*mu)/np.math.factorial(s))*(np.exp(-(x-))) for p in range(1)]) for s in range(4)])
    return f

def MultiGauss(x,mu,lam,sig,Q0,Q1,n,q):
    f = sum([sum([np.exp(-lam*p)*(lam*p)**s/np.math.factorial(s)*mu**p*np.exp(-mu)/(np.math.factorial(p)*sig*np.sqrt(2.*np.pi))*np.exp(-(x-(p+s)*Q1-Q0)**2/(2.*sig**2)) for p in range(n)]) for s in range(q)])
    return f

#params = [1.,1.,-3.]; #parameters for a single gaussian                                               
params=[1.,1.,1.,2.,-3.,0.]; #parameters for the sum of two gaussians
xdata=np.arange(-6,6,0.01)
#ydata = np.array([GaussSum(x,*params) for x in xdata])
#popt,pcov = curve_fit(SiPM_Fit,xdata,ydata)

fvals = np.linspace(0,40,1000)
#f2 = SiPM_Fit(fvals,lam=0.5,mu=3,sig0=1,sig1=.5,g=1,x0=1)
f = MultiGauss(fvals,mu=3,lam=.5,sig=1,Q0=2,Q1=3,n=5,q=1)

plt.figure()
#plt.plot(xdata,SiPM_Fit(xdata,*popt))
plt.plot(fvals,f,'r')
plt.show()
