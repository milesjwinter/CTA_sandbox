import numpy as np
import matplotlib.pyplot as plt


Nasic = 0
Nchannel = 14
Nblock = 256
Nphase = 12
#Load correct transfer function for each set of samples
#data_mono = np.load("Test_TF/TF_mono_ASIC"+str(Nasic)+"CH"+str(Nchannel)+"BN"+str(Nblock)+"BP"+str(Nphase)+".npz")['arr_0']
#data_lin = np.load("Test_TF/TF_lin_ASIC"+str(Nasic)+"CH"+str(Nchannel)+"BN"+str(Nblock)+"BP"+str(Nphase)+".npz")['arr_0']

data_sync = np.load('Test_TF/TF_sync_ASIC0CH14BN0BP4.npz')['arr_0']
ADC_sync = np.array(data_sync[:,0],dtype=int)
mV_sync = np.array(data_sync[:,1:])

#ADC_mono = np.array(data_mono[:,0],dtype=int)
#mV_mono = np.array(data_mono[:,1:])

#ADC_lin = np.array(data_lin[:,0],dtype=int)
#mV_lin = np.array(data_lin[:,1:])

low_tf, mean_tf, up_tf = np.percentile(mV_sync,[16.,50.,84.],axis=1)*1.e-3
print np.argmin(np.absolute(mean_tf-1.5))
print mean_tf[311],'  ',mean_tf[311]-low_tf[311],'   ',mean_tf[311]-up_tf[311]
print ADC_sync[311]

z = np.polyfit(mean_tf[800:2800], ADC_sync[800:2800], 1)
print z
points = np.arange(0,2,1)
plt.figure()
#plt.plot(low_tf,ADC_sync,color='gray',linestyle='dashed',lw=2)
#plt.plot(mean_tf,ADC_sync,'r',lw=2,label='mean')
#plt.plot(up_tf,ADC_sync,color='gray',linestyle='dashed',lw=2,label='mean$\pm 1\sigma$')
#plt.plot(mean_tf,mean_tf*z[0]+z[1],'k',lw=1,label='linear fit')
#plt.plot(mean_tf[311],ADC_sync[311],'bo',lw=2,label='$V_{ped}=.7V$')
#plt.plot(points,points*z[0]+z[1],label='linear fit')
for i in range(128):
    plt.plot(mV_sync[:,i]*1.e-3,ADC_sync,lw=0.8)
    #plt.plot(mV_mono[:,i]*1.e-3,ADC_mono,label='Monotone Spline')
    #plt.plot(mV_lin[:,i],ADC_lin,label='Linear Interpolation')
plt.xlabel('Input DC Voltage (V)')
plt.ylabel('Measured ADC Counts')
plt.legend()
plt.show()

'''
plt.figure()
for p in range(4):
    phase = p*8+4
    data_mono = np.load("Test_TF/TF_mono_ASIC"+str(Nasic)+"CH"+str(Nchannel)+"BN"+str(Nblock)+"BP"+str(phase)+".npz")['arr_0']
    ADC_mono = np.array(data_mono[:,0],dtype=int)
    mV_mono = np.array(data_mono[:,1:])
    plt.plot(mV_mono[:,22]*1.e-3,ADC_mono,label='%s ns'%phase)
plt.xlabel('Input DC Voltage (V)')
plt.ylabel('Measured ADC Counts')
plt.legend(title='Trigger Phase')
plt.show()
'''
