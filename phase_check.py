import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import sys

def RMS(x):
    return np.sqrt(np.sum(x**2,axis=1)/len(x[0,:]))

#Run information
#runs = np.arange(316170,316180,1)
#runs = [317439,317440,317443,317445,317446,317448,317450,317452,317454,317455,317456,317457,317458,317459]
#runs = [317467,317468,317469,317470,317471,317472,317473,317474]
#runs = [317115,317118,317123,317127,317129,317130,317133,317138,317141,317144,317148,317152,317153,317154,317155]
#runs = np.arange(317516,317536,1)
#runs = np.array([317433,317434,317435,317437])
#runs = np.arange(317505,317514,1)
#runs = np.arange(317484,317491,1)
runs = np.array([317467])
#runs = np.arange(319418,319427,1)
#runs = np.append(runs,np.arange(317484,317491,1))
#runs = np.append(runs,np.arange(317505,317515,1))
#runs = np.append(runs,np.arange(317516,317536,1))
#runs = np.arange(319462,319472,1)
#runs = np.arange(319518,319558,1)
#runs = np.arange(319558,319568,1)
#runs = np.arange(319518,319568,1)
#runs = np.arange(319428,319435,1)
#runs = np.arange(319172,319182,1)
Nblocks = 2
Ncells = Nblocks*32

all_blocks = np.array([])
all_phases = np.array([])
cal_phases = np.array([])

#Loop through each run
for i, run_num in enumerate(runs):
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load samples
            data = np.load("run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
            run = data['run']
            asic = data['asic']
            channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            del data

            if run_num < 319558:
                all_blocks = np.append(all_blocks,block)
                all_phases = np.append(all_phases,phase)
            else:
                cal_phases = np.append(cal_phases,phase)
           

#load pedestal waveforms
Ped_dir = "run_files"
Ped_infile = "/baseline_async_hv_off_avg_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
data = np.load(Ped_dir+Ped_infile)
ped_block1 = np.array(data['block'],dtype=int)
ped_phase1 = np.array(data['phase'],dtype=int)
ped_samples1 = np.array(data['baseline'],dtype='float32')
del data


#load pedestal waveforms
Ped_infile = "/baseline_async_LED_avg_new_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
data = np.load(Ped_dir+Ped_infile)
ped_block2 = np.array(data['block'],dtype=int)
ped_phase2 = np.array(data['phase'],dtype=int)
ped_samples2 = np.array(data['baseline'],dtype='float32')
del data


'''
#load pedestal waveforms
Ped_dir = "run_files"
Ped_infile = "/baseline_LED_avg_newest_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
#Ped_infile = "/baseline_LED_phase_avg_newest_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
#Ped_infile = "/baseline_TF_avg_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
data = np.load(Ped_dir+Ped_infile)
ped_block1 = np.array(data['block'],dtype=int)
ped_phase1 = np.array(data['phase'],dtype=int)
ped_samples1 = np.array(data['baseline'],dtype='float32')
del data

#load pedestal waveforms
#Ped_infile = "/baseline_LED_async_avg_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
Ped_infile = "/baseline_LED_phase_avg_newest_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
data = np.load(Ped_dir+Ped_infile)
ped_block2 = np.array(data['block'],dtype=int)
ped_phase2 = np.array(data['phase'],dtype=int)
ped_samples2 = np.array(data['baseline'],dtype='float32')
del data
'''
start = 42-4
stop = 42+4
charge1 = np.sum(ped_samples1[:,start:stop],axis=1)
#charge2 = np.sum(ped_samples2[:,start:stop],axis=1)
#print charge2[:30]
times = np.arange(0,64,1)
plt.figure(1,figsize=(14,7))
for i in range(0,32):
    #shift = ped_phase1[i]
    #times = np.arange(shift,shift+64,1)
    if (ped_phase1[i]+1)%4>0:
        if ped_phase1[i]>26:# and ped_phase1[i]<16:
            p = plt.plot(times,ped_samples1[i],lw=0.8,linestyle='dashed',label='phase %s'%(int(ped_phase1[i])))
        
            plt.plot(times,ped_samples2[i],lw=0.8,color=p[0].get_color()) #,label='LED: phase %s'%(int(ped_phase2[i])))
        #plt.plot(times,ped_samples1[i]-np.mean(ped_samples1[i,:24]),label='block %s, phase %s'%(int(ped_block1[i]),int(ped_phase1[i])))
    #plt.plot(times,ped_samples1[i]-np.mean(ped_samples1[i,:24]),label='block %s, phase %s'%(int(ped_block1[i]),int(ped_phase1[i])))
    #plt.plot(times,ped_samples2[i]-np.mean(ped_samples2[i,:64]),label='block %s, phase %s'%(int(ped_block2[i]),int(ped_phase2[i])))
        #plt.plot(times,ped_samples2[i],label='block %s, phase %s'%(int(ped_block2[i]),int(ped_phase2[i])))
        #plt.plot(times,ped_samples2[i],'b')
#plt.plot(times,ped_samples2[10]-np.mean(ped_samples2[10,:24]),'k',lw=2.0,label='avg phase %s'%(int(ped_phase2[10])))
plt.xlabel('Time (ns)')
plt.ylabel('ADC Counts')
#plt.ylabel('mV')
#plt.title('TF Calibrated Waveforms')
plt.title('Pedestal Subtracted Waveforms')
plt.xlim([0,63])
#plt.ylim([-10,10])
plt.xticks(np.arange(0,64+2, 8.0))
plt.legend(ncol=2,fontsize=10)
plt.show()
'''
#plt.figure(1,figsize=(14,7))
plt.figure()
plt.hist(charge1[charge1>0],bins=np.arange(0,300,4),edgecolor='k')
#plt.xlabel('Charge (mV s)',fontsize=20)
plt.xlabel('Charge (ADC s)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.title('TF Calibrated',fontsize=24)
plt.title('Pedestal Subtracted',fontsize=24)
#plt.xticks(np.arange(0,Ncells+2, 32.0))
plt.minorticks_on()
#plt.legend(bbox_to_anchor=(0, 0, 1, 1),ncol=2)
plt.show()
'''

plt.figure()
plt.hist(all_phases,bins=32,normed=True, label='HV On',alpha=0.8)
plt.hist(cal_phases,bins=32,normed=True,label='HV Off',alpha=0.8)
plt.xlabel('Phase',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.title('Phase Distribution',fontsize=24)
plt.legend()
plt.show()

