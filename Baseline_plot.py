import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import time
import sys
from matplotlib import gridspec

start_time = time.time()

def RMS(x):
    return np.sqrt(np.sum(x**2,axis=1)/len(x[0,:]))

#Run information
Nasic = 0
Nchannel = 14
#run_num= 319180
run_num = 319425
#run_num = 316151
#run_num = 314499
vped = 1106.
Ncells = 64
#Nruns = 316151
duration = 1200

#Load samples
print "Loading Waveforms ...."
'''
#data = np.load("run_files/sampleFileAllBlocks_run316151ASIC0CH14.npz")
data = np.load("base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(0)+"_CH"+str(14)+".npz")
#data = np.load('run_files/baseline_avg_LEDASIC0CH14.npz')
#data = np.load("run_files/sampleFileAllBlocks_run314510ASIC0CH14.npz")
#data = np.load("run_files/sampleFileAllBlocks_run317111ASIC0CH14.npz")
#raw_samples = data['baseline']
raw_samples = data['samples']
#raw_event = data['event']
#baseline = np.percentile(raw_samples,50.,axis=0)
baseline = np.mean(raw_samples,axis=0)
corr_baseline = baseline-np.mean(raw_samples[:,:20])
#del data
'''
data = np.load("run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
#data = np.load("base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(0)+"_CH"+str(14)+".npz")
samples = data['samples']
block = data['block']
phase = data['phase']
#samples = data['baseline']
#event = data['event']
times = np.arange(0,Ncells,1)
#samp_avg = np.mean(samples)
'''
corr_samples = np.zeros((len(event),Ncells))
alt_corr_samples = np.zeros((len(event),Ncells))
#base_samples = np.zeros((len(raw_event),Ncells))
ped_samples = np.zeros((len(event),Ncells))
for i in range(len(event)):
     alt_corr_samples[i,:] = samples[i,:]-samp_avg-corr_baseline
     corr_samples[i,:] = samples[i,:]-np.mean(samples[i,:20])
     ped_samples[i,:] = corr_samples[i,:]-np.mean(corr_samples[i,:64])
#for i in range(len(raw_event)):
#    base_samples[i,:] = raw_samples[i,:]-baseline
'''
#max_val = np.amax(corr_samples[:,:],axis=1)

print "Plotting Waveforms ...."
#plt.figure()
#plt.plot(event[:50],max_val[:50],'o')
#plt.show()
#print np.amin(corr_samples)
print block[0]
print phase[0]
plt.figure(1,figsize=(14,7))
for i in range(25):
    b = block[i]
    p = phase[i]
    index = b*32+p
    #if block[i]==180:
    #    if phase[i]==14:
    #        plt.plot(times,samples[i],'r', lw=1.,alpha=0.8)
    #plt.plot(times,samples[i], lw=1.,alpha=0.8,label='block %s, phase %s'%(int(block[i]),int(phase[i])))
    #plt.plot(times,samples[i]-raw_samples[index],'b', lw=1.,alpha=0.8)
    #new_samples = samples[i]-raw_samples[index]
    #new_ped = np.mean(new_samples[:24])
    #fixed_samples = new_samples - new_ped
    plt.plot(times,samples[i],'r', lw=1.,alpha=0.8)
    #plt.plot(times,fixed_samples,'r', lw=1.,alpha=0.8)

    #if np.amax(corr_samples[i])>23 and np.amax(corr_samples[i])<26:
    #if np.argmax(corr_samples[i])>=152 and np.argmax(corr_samples[i])<=153:
    #if np.argmax(corr_samples[i])>=152:
    #    plt.plot(times,alt_corr_samples[i,:Ncells],'g', lw=0.5,alpha=0.5)
    #if np.argmax(corr_samples[i])>=153:
    #    plt.plot(times,alt_corr_samples[i,:Ncells],'b', lw=0.5,alpha=0.5)
    #    #plt.plot(times, alt_corr_samples[i,:Ncells],'b',lw=.5,alpha=0.5)
    #plt.plot(times,samples[i,:]-np.mean(samples),'b',alpha=0.7)
    #plt.plot(times,raw_samples[i,:]-np.mean(raw_samples),'r',alpha=0.7)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Meas. ADC Counts',fontsize=20)
#plt.xlim([0,Ncells])
plt.xlim([0,64])
#plt.xlim([128,192])
#plt.title('Pedestal Subtracted Waveforms',fontsize=24)
plt.title('Calibration: Raw Waveforms',fontsize=24)
#plt.legend()
#plt.xticks(np.arange(0, Ncells+2, 32.0))
#plt.minorticks_on()
plt.show()
'''
plt.figure(2,figsize=(14,7))
plt.plot(times,np.mean(corr_samples,axis=0),'b',alpha=0.7)
plt.plot(times,np.mean(base_samples,axis=0),'r',alpha=0.7)
plt.plot(times,np.mean(alt_corr_samples,axis=0),'g',alpha=0.7)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Meas. ADC Counts',fontsize=20)
plt.xlim([0,Ncells])
#plt.xlim([128,192])
plt.title('Raw Waveforms',fontsize=24)
plt.xticks(np.arange(0, Ncells+2, 32.0))
plt.minorticks_on()
plt.show()
'''
'''
std = np.std(corr_samples[:,:128],axis=1)
plt.figure(2,figsize=(14,7))
for i in range(len(std)):
    if std[i]>40:
        plt.plot(np.arange(0,256,1),corr_samples[i,:],label='Event: %s' % event[i])
#plt.plot(np.arange(0,128,1),np.mean(base_samples[:,:128],axis=0),label='HV Off')
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Measured ADC Counts',fontsize=20)
#plt.xlim([0,Ncells])
#plt.xlim([0,128])
plt.title('Calibrated Waveforms: Run %s' % run_num,fontsize=24)
plt.xticks(np.arange(0, 256+2, 32.0))
plt.legend()
plt.minorticks_on()
plt.show()
'''
'''
fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax0 = plt.subplot(gs[0])
ax0.hist(np.mean(corr_samples[:,:128],axis=1),bins=np.arange(-12,6,0.25),edgecolor='k',label='HV On (Run 317530)',normed=True,alpha=1)
ax0.hist(np.mean(base_samples[:,:128],axis=1),bins=np.arange(-12,6,0.25),edgecolor='k',label='HV Off',normed=True,alpha=.75)
ax0.set_xlabel('Mean',fontsize=20)
ax0.set_ylabel('Counts',fontsize=20)
ax0.minorticks_on()
plt.legend()
plt.title('First 4 Blocks',fontsize=24)
ax1 = plt.subplot(gs[1])
ax1.hist(np.std(corr_samples[:,:128],axis=1),bins=np.arange(1,12,0.25),edgecolor='k',label='HV On (Run 317530)',normed=True,alpha=1)
ax1.hist(np.std(base_samples[:,:128],axis=1),bins=np.arange(1,12,0.25),edgecolor='k',label='HV Off',normed=True,alpha=.75)
#ax1.hist(RMS(corr_samples[:,:128]),bins=np.arange(0,8,0.25),edgecolor='k',label='HV On (Run 316175)',normed=True,alpha=1)
#ax1.hist(RMS(base_samples[:,:128]),bins=np.arange(0,8,0.25),edgecolor='k',label='HV Off',normed=True,alpha=.75)
ax1.set_xlabel('Standard Deviation',fontsize=20)
ax1.set_ylabel('Counts',fontsize=20)
ax1.minorticks_on()
plt.legend()
plt.tight_layout()
plt.show()
'''
'''
plt.figure(2,figsize=(16,8))
plt.plot(times,baseline)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Measured ADC Counts',fontsize=20)
plt.xlim([0,Ncells])
plt.ylim([600,780])
plt.title('Baseline Waveform',fontsize=24)
plt.xticks(np.arange(min(times), max(times)+2, 32.0))
plt.minorticks_on()
plt.show()

plt.figure(3,figsize=(16,8))
for i in range(len(event)):
    plt.plot(times,corr_samples[i,:])
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Measured ADC Counts',fontsize=20)
plt.xlim([0,Ncells])
#plt.ylim([600,780])
plt.title('Deviation from Baseline',fontsize=24)
plt.xticks(np.arange(min(times), max(times)+2, 32.0))
plt.minorticks_on()
plt.show()
'''
'''
#plt.figure(4,figsize=(16,8))
fig, ax = plt.subplots()
plt.subplot(221)
plt.hist(np.arange(0,256,1),bins=np.arange(0,256,1),weights=np.std(corr_samples,axis=0))
plt.ylabel('RMS')
plt.subplot(223)
for i in range(len(event)):
    plt.plot(times,corr_samples[i,:])
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('ADC Counts')
plt.xlim([0,Ncells])
plt.minorticks_on()
plt.subplot(224)
plt.hist(corr_samples.flatten(),bins=np.arange(-15,15,1),orientation='horizontal')
plt.xlabel('Counts')
#plt.xlim([0,Ncells])
#plt.ylim([600,780])
#plt.title('Deviation From Baseline Waveform',fontsize=24)
#plt.xticks(np.arange(min(times), max(times)+2, 32.0))
plt.minorticks_on()
plt.show()
'''
'''
fig = plt.figure(figsize=(16, 8)) 
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1],height_ratios=[1,3]) 
ax0 = plt.subplot(gs[0])
ax0.hist(np.arange(0,256,1),bins=np.arange(0,256,1),histtype='step',weights=np.std(corr_samples,axis=0))
ax0.set_ylabel('RMS',fontsize=20)
ax0.set_xlim([0,Ncells-1])
ax0.minorticks_on()
ax0.set_xticks(np.arange(min(times), max(times)+2, 32.0))
ax2 = plt.subplot(gs[2])
for i in range(len(event)):
    ax2.plot(times,corr_samples[i,:])
ax2.set_xlabel('Time (ns)',fontsize=20)
ax2.set_ylabel('ADC Counts',fontsize=20)
ax2.set_xlim([0,Ncells-1])
ax2.set_xticks(np.arange(min(times), max(times)+2, 32.0))
ax2.minorticks_on()
ax3 = plt.subplot(gs[3])
ax3.hist(corr_samples.flatten(),bins=np.arange(-15,15,1),orientation='horizontal')
ax3.set_xlabel('Counts',fontsize=20)
ax3.set_ylim([-15,20])
#plt.ylim([600,780])
#plt.title('Deviation From Baseline Waveform',fontsize=24)
ax3.set_xticks(np.arange(0, 30001, 10000))
ax3.minorticks_on()
plt.tight_layout()
plt.show()
'''
