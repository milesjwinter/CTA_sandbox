import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

#Run information
run = 313369
asic = 0
channel = 0
vped = 1106.

#Load samples
data = np.loadtxt("run_files/sampleFileLarge_run"+str(run)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
ievt = np.array(data[:,2]) 
blocknum = np.array(data[:,3]) 
blockphase = np.array(data[:,4])
samples = np.array(data[:,5:],dtype=float)
times = np.arange(0,64,1)

#Load baseline
baseline = 0.0
data = np.loadtxt("run_files/flasher_av_run"+str(run)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
for i in range(1,len(data[:,0])):
    if data[i,4]>0.0:
        baseline = np.array(data[i,4:])

#Correct samples: subtract baseline
corr_samples = samples-baseline

'''
clean_samples = np.array([])
for i in range(len(corr_samples[:,0])):
    #if np.amin(corr_samples[i,:]) >= -20.:
    clean_samples = np.append(clean_samples,corr_samples[i,:])
#plt.hist(clean_samples,bins=100,)
#plt.yscale('log')
#plt.show()
'''

limits = np.percentile(corr_samples, [2.5,50.,97.5,99.85],axis=0)
print '16%= ',limits[0], ',  50%= ', limits[1], ',  99%= ',limits[2]


print 'job time: ', time.time() - start_time

'''
plt.figure(2)
for i in range(len(corr_samples[:,0])):
    if np.amax(corr_samples[i,:]) >= 40. and np.amin(corr_samples[i,:]) >= -5.:
        plt.plot(times,corr_samples[i,:])
plt.xlim([0,63])
plt.show()
'''

plt.figure(3)
for i in range(len(corr_samples[:,0])):
    lower = corr_samples[i,:]-limits[0]
    upper = corr_samples[i,:]-limits[2]
    if np.amin(lower) >= 0. and np.amax(upper) >= 0.:
        plt.plot(times,corr_samples[i,:])
#plt.plot(times,limits[0],'k',lw=2.0,label='1%')
plt.plot(times,limits[1],'k',lw=2.0,label='50%')
plt.plot(times,limits[2],'k',lw=2.0,label='2sigma')
plt.plot(times,limits[3],'k',lw=2.0,label='3sigma')
plt.legend()
plt.xlim([0,63])
plt.show()

