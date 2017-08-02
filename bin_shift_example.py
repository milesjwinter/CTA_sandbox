import matplotlib.pyplot as plt
import numpy as np

#Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2.*sigma**2))

def linear_function(x, m, b):
    return m*x+b

x = np.linspace(0,20,1000)
noise = np.random.normal(0,.01,size=1000)
gauss = gauss_function(x,a=1.,x0=10,sigma=1)+noise
line = linear_function(x,m=0.02,b=-0.4)+noise
gauss_line = gauss+line

print np.amax(gauss)
print np.amax(gauss_line)

plt.figure()
plt.plot(x,gauss)
plt.plot(x,gauss_line)
#plt.axvline(x[np.argmax(gauss)],lw=0.5,color='blue')
#plt.axvline(x[np.argmax(gauss_line)],lw=0.5,color='orange')
plt.show()
