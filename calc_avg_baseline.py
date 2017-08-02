import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import sys


#Run information
#runs = np.arange(314515,314532,1)
#runs = np.arange(314536,314549,1)
#runs = np.arange(318767,318775,1)
#runs = np.arange(319164,319172,1)
#runs = np.arange(319210,319223,1)
#runs = np.arange(319252,319272,1)
#runs = np.arange(319272,319292,1)
#runs = np.arange(319174,319182,1)
#runs = np.arange(319420,319428,1)
#runs  = np.arange(319452,319457,1)
#runs = np.arange(319558,319568,1)
#runs = np.arange(319418,319428,1)
#runs = np.arange(319428,319435,1)
#runs = np.arange(319162,319172,1)
#runs = np.arange(319172,319182,1)
#runs = np.arange(314501,314511,1)
#runs = [314540]
#runs = np.arange(314536,314549,1)
runs = np.arange(320255,320257,1)
Nblocks = 8
Ncells = Nblocks*32
nPhases = 4

avg_baseline = np.zeros((512,nPhases,Ncells))
avg_phase = np.zeros((nPhases,Ncells))
counts = np.zeros((512,nPhases))
phase_counts = np.zeros(nPhases)
asic = []
channel = []
block = []
phase = []
#Loop through each run
for i, run_num in enumerate(runs):
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load samples
            #infile = "run_files/flasher_av_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            infile = "run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            #infile = "calib_waveforms/calib_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile = "base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            data = np.load(infile)
            event = data['event']
            asic = data['asic']
            channel = data['channel']
            block = data['block']
            phase = data['phase']
            #samples = np.array(data['baseline'],dtype='float32')
            samples = np.array(data['samples'],dtype='float32')
            del data
            
            for q in range(len(event)):
                b = block[q]
                #p = phase[q]
                #p = (phase[q]-4)/8
                p = phase[q]/8
                #if np.amax(samples[q,:])<=10. and np.amin(samples[q,:])>=-10.:
                #if np.std(samples[q,:24])<5.:
                if np.amin(samples[q])>100.:
                    avg_baseline[b,p,:] += samples[q,:]#-np.mean(samples[q,:])
                    counts[b,p] += 1.
                    #corr_samples = samples[q,:]-np.mean(samples[q,:20])
                    #if np.amax(corr_samples[:20])<=5. and np.amin(corr_samples[:20])>=-5.:
                    #    avg_phase[p,:] += corr_samples
                    #    phase_counts[p] += 1.
            '''
	    for b in range(512):
		for p in range(nPhases):
                    index = b*nPhases+p
		    if block[index] == b:
			if phase[index] == p:
			    if samples[index,0] > 0.:
				avg_baseline[b,p,:] += samples[index,:]
				counts[b,p] += 1.
            '''
baseline = np.zeros((512*nPhases,Ncells))
#baseline = np.zeros((nPhases,Ncells))
block = np.array([])
phase = np.array([])

for b in range(512):
    for p in range(nPhases):
        block = np.append(block,b)
        phase = np.append(phase,p)
        if counts[b,p]>0.:
            #print counts[b,p]
            index = b*nPhases+p
            baseline[index,:] = avg_baseline[b,p,:]/counts[b,p]
'''

for p in range(nPhases):
        #block = np.append(block,b)
        phase = np.append(phase,p)
        if phase_counts[p]>0.:
            #print counts[b,p]
            index = b*nPhases+p
            baseline[p,:] = avg_phase[p,:]/phase_counts[p]
'''
outfile = "run_files/baseline_8_block_sync_avg_1556_"+"ASIC"+str(asic[0])+"CH"+str(channel[0])+".npz"
np.savez_compressed(outfile,asic=asic,channel=channel,block=block,phase=phase,baseline=baseline,phase_counts=phase_counts)


