import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import sys

def RMS(x):
    return np.sqrt(np.sum(x**2,axis=1)/len(x[0,:]))

#Run information
#runs = np.arange(316170,316180,1)
#runs = np.arange(318765,318775,1)
#runs = np.arange(318745,318765,1)
#runs  = np.arange(319253,319272,1)
#runs = np.arange(314490,314500,1)
#runs = np.arange(314534,314548,1)
runs = np.arange(320235,320244,1)
runs = np.append(runs,np.arange(320245,320247,1))
#runs = [320246,320254]
#runs = np.arange(320247,320251,1)
#runs = [314517]
#runs = np.append(runs,np.arange(314513,314532,1))
#runs = [317439,317440,317443,317445,317446,317448,317450,317452,317454,317455,317456,317457,317458,317459]
#runs = [317467,317468,317469,317470,317471,317472,317473,317474]
#runs = [317115,317118,317123,317127,317129,317130,317133,317138,317141,317144,317148,317152,317153,317154,317155]
#runs = np.arange(317516,317536,1)
#runs = np.arange(317520,317536,1)
#runs = np.array([317433,317434,317435,317437])
#runs = np.append(runs,np.arange(317505,317514,1))
#runs = np.arange(317484,317491,1)
Nblocks = 8
Ncells = Nblocks*32

#load and calculate baseline waveform
data = np.load("run_files/sampleFileAllBlocks_run317111ASIC0CH14.npz")
#data = np.load("run_files/sampleFileAllBlocks_run317054ASIC0CH14.npz")
#data = np.load("run_files/sampleFileAllBlocks_run316151ASIC0CH14.npz")
raw_samples = data['samples']
baseline = np.mean(raw_samples,axis=0)
corr_baseline = baseline-np.mean(raw_samples[:,:64])
del data

max_base_amp = np.array([])
pulse_position = np.array([])
phases = np.array([])
mean_waveform = np.zeros((len(runs),Ncells))
mean_waveform1 = np.zeros((len(runs),Ncells))
mean_waveform2 = np.zeros((len(runs),Ncells))
mean_waveform3 = np.zeros((len(runs),Ncells))
mean_waveform4 = np.zeros((len(runs),Ncells))
plt.figure(1,figsize=(12,5))
#Loop through each run
for i, run_num in enumerate(runs):
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load samples
            #data = np.load("run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
            data = np.load("base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            #data = np.load("calib_waveforms/calib_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            run = data['run']
            asic = data['asic']
            channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            del data

            #initialize a few variables and temp arrays
            Nevents = len(event)         #num of events in samples
            print Nevents
            #Correct samples with pedestal subtraction in mV.
            #pedestal = np.mean(samples[:,:64],axis=1)
            #pedestal = pedestal.reshape(len(pedestal),1)
            #corr_samples = samples-pedestal
            #corr_samples = samples-baseline
            corr_samples = samples
            '''
	    std = np.std(corr_samples[:,:128],axis=1)
            label = 'plots/baseline_studies/large_pulses_%s.png' % run_num
	    plt.figure(2,figsize=(14,7))
	    for i in range(len(std)):
	        if std[i]>40:
		    plt.plot(np.arange(0,256,1),corr_samples[i,:],label='Event: %s' % event[i])
	    plt.xlabel('Time (ns)',fontsize=20)
	    plt.ylabel('Measured ADC Counts',fontsize=20)
	    plt.title('Calibrated Waveforms: Run %s' % run_num,fontsize=24)
	    plt.xticks(np.arange(0, 256+2, 32.0))
	    plt.legend()
	    plt.minorticks_on()
            plt.savefig(label)
            plt.show()
            '''

            #max_base_amp = np.append(max_base_amp,np.amax(corr_samples[:,:128],axis=1))
            #pulse_position = np.append(pulse_position,145+np.argmax(corr_samples[:,145:160],axis=1))
            #label = 'plots/baseline_studies/compare_amplitude_%s.png' % run_num
            #plt.figure(2,figsize=(14,7))
            #plt.plot(pulse_position,max_base_amp,'k*')
            #plt.hist2d(pulse_position, max_base_amp,bins=(np.arange(144.5,160.5,1),np.arange(0,300,10)),cmap=plt.cm.jet)
            #plt.xlabel('Time (ns)',fontsize=20)
            #plt.ylabel('Measured ADC Counts',fontsize=20)
            #plt.title('Max Baseline Amplitude vs Pulse Position: Run %s' % run_num,fontsize=24)
            #plt.xticks(np.arange(0, 256+2, 32.0))
            #plt.legend()
            #plt.colorbar()
            #plt.minorticks_on()
            #plt.savefig(label)
            #plt.show()

            #plt.plot(np.arange(0,Ncells,1),np.mean(corr_samples[:,:128],axis=0),label='Run '+str(run_num))
            index = run_num-runs[0]
            temp_max = np.zeros(len(event))
            good_index = phase==4
            mean_waveform[i,:] = np.mean(corr_samples,axis=0)
            mean_waveform1[i,:] = np.mean(corr_samples[phase==4],axis=0)
            mean_waveform2[i,:] = np.mean(corr_samples[phase==12],axis=0)
            mean_waveform3[i,:] = np.mean(corr_samples[phase==20],axis=0)
            mean_waveform4[i,:] = np.mean(corr_samples[phase==28],axis=0)
            #mean_waveform[i,:] = np.mean(corr_samples[:,:Ncells],axis=0)
            #for i in range(len(event)):
            #    temp_max[i] = np.argmax(corr_samples[i,:])
            #pulse_position = np.append(pulse_position,np.argmax(corr_samples,axis=1))
            #phases = np.append(phases,phase[:])
            phases = np.append(phases,phase)
print mean_waveform1.shape

for i in range(len(runs)):
    #plt.plot(np.arange(0,Ncells,1),mean_waveform[i,:]-np.mean(mean_waveform[i,:],axis=0),label='Run '+str(run_num))
    plt.plot(np.arange(0,Ncells,1),mean_waveform[i,:],label='Run '+str(runs[i]))
plt.xlabel('Time (ns)',fontsize=20)
#plt.ylabel('Measured ADC Counts',fontsize=20)
#plt.ylabel('ADC Counts',fontsize=20)
plt.ylabel('Amplitude (mV)',fontsize=20)
#plt.xlim([0,Ncells])
plt.xlim([0,Ncells])
#plt.ylim([-10,15])
#plt.title('Average Pedestal Subtracted Waveforms',fontsize=24)
#plt.title('Average DC Calibrated Waveforms',fontsize=24)
plt.xticks(np.arange(0,Ncells+2, 32.0))
#plt.minorticks_on()
plt.legend(bbox_to_anchor=(0, 0, 1, 1),ncol=1,fontsize=12)
plt.show()

plt.figure(2,figsize=(12,5))
plt.plot(np.arange(0,Ncells,1),np.mean(mean_waveform1,axis=0),label='Phase=4ns ')
plt.plot(np.arange(0,Ncells,1),np.mean(mean_waveform2,axis=0),label='Phase=12ns ')
plt.plot(np.arange(0,Ncells,1),np.mean(mean_waveform3,axis=0),label='Phase=20ns ')
plt.plot(np.arange(0,Ncells,1),np.mean(mean_waveform4,axis=0),label='Phase=28ns ')
plt.xlabel('Time (ns)',fontsize=20)
#plt.ylabel('Measured ADC Counts',fontsize=20)
#plt.ylabel('ADC Counts',fontsize=20)
plt.ylabel('Amplitude (mV)',fontsize=20)
#plt.xlim([0,Ncells])
plt.xlim([0,Ncells])
#plt.ylim([-10,15])
#plt.title('Average Pedestal Subtracted Waveforms',fontsize=24)
#plt.title('Average DC Calibrated Waveforms',fontsize=24)
plt.xticks(np.arange(0,Ncells+2, 32.0))
#plt.minorticks_on()
plt.legend(bbox_to_anchor=(0, 0, 1, 1),ncol=1,fontsize=12)
plt.show()

'''
plt.hist(pulse_position,bins=np.arange(120,160,1))
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.title('Position of Pulse Maximum')
plt.show()
'''
plt.figure(2)
plt.hist(phases, bins=np.arange(0,33,1),edgecolor='k', lw=0.8)
plt.title('Phases: Calibration Data',fontsize=24)
plt.xlabel('Phase',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.xlim([0,33])
plt.xticks(np.arange(0,33, 1))
plt.show()

'''
print pulse_position.shape
print max_base_amp.shape
H, xedges, yedges = np.histogram2d(pulse_position, max_base_amp, bins=(np.arange(145,160,1), np.arange(0,300,50)),normed=True)
H = H.T
X, Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X,Y,H)
plt.colorbar()
plt.show()

plt.hist2d(pulse_position, max_base_amp,cmap=plt.cm.jet)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Measured ADC Counts',fontsize=20)
plt.title('Max Baseline Amplitude vs Pulse Position: Run %s' % run_num,fontsize=24)
#plt.xticks(np.arange(0, 256+2, 32.0))
#plt.legend()
#plt.colorbar()
plt.minorticks_on()
#plt.savefig(label)
plt.show()
'''
