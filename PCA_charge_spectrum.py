import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA, FastICA
import time
import sys

start_time = time.time()

#Run information
runs = np.arange(316170,316180,1)
points = np.arange(0,1200,1)
hist_array = np.zeros((len(points),len(runs)))
N = 3
#Loop through each run
for i in range(len(runs)):
    print "Loading calibrated waveforms: Run ",runs[i]
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):
            data = np.load("charge_spectrum_files/stored_charge_histogram_Run"+str(runs[i])+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            hist_array[:,i] = data['charge_hist']
            
# Compute PCA
pca = PCA(n_components=N)
H = pca.fit_transform(hist_array)  # Reconstruct signals based on orthogonal components
new_samples = H[:,0]
new_samples = np.transpose(new_samples)

# Compute ICA
ica = FastICA(n_components=N)
S = ica.fit_transform(hist_array)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix

'''
plt.figure()
models = [hist_array, H]
names = ['Original Spectrum',
     'PCA recovered Spectrum']
colors = ['red', 'steelblue', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 1, ii)
    plt.title(name)
    for sig in zip(model.T):
        plt.plot(sig)
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
'''
plt.figure(1,figsize=(16,8))
for i in range(1,2):
    plt.plot(points,S[:,i])
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.title('Charge Spectrum',fontsize=24)
plt.show()


