"""
Miles Winter
Run raw waveforms through pedestal subtraction and write out to file
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

start_time = time.time()

#Run information
#runs = np.arange(318765,318775,1)
#runs = np.arange(319147,319182,1)
#runs = np.arange(319208,319223,1)
#runs = np.arange( 319172,319182,1)
#runs = np.arange(319253,319272,1)
#runs = np.arange(314515,314533,1)
#runs = np.arange(319272,319292,1)
#runs = np.arange(314534,314549,1)
#runs = np.append(runs,np.arange(314513,314533,1))
#runs = np.arange(314490,414500,1)
#runs = np.arange(316170,316180,1)
#runs = np.arange(319462,319472,1)
#runs = np.arange(319518,319558,1)
#runs = np.arange(319428,319435,1)
#runs = np.arange(319172,319182,1)
#runs = np.arange(319252,319272,1)
#runs = np.arange(314512,314549,1)
runs = [320246]
#runs = np.arange(320235,320244,1)
#runs = np.append(runs,np.arange(320245,320247,1))
#runs = np.arange(320247,320251,1)
Nblocks = 8
Ncells = Nblocks*32
nPhases = 4

#specify transfer function and data directory
Ped_dir = "run_files"
Data_dir = "run_files"
#Loop through each run
for Nrun in runs:
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):
            #create output file for calibrated samples
            outfile = "base_cal_runs/base_cal_samples_Run"+str(Nrun)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            
            print 'Processing Run ',Nrun,' Asic ',Nasic,' Channel ',Nchannel
            #Load samples
            data = np.load(Data_dir+"/sampleFileAllBlocks_run"+str(Nrun)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
            run = np.array(data['run'],dtype=int)
            asic = np.array(data['asic'],dtype=int)
            channel = np.array(data['channel'],dtype=int)
            event = np.array(data['event'],dtype=int)
            block = np.array(data['block'],dtype=int)
            phase = np.array(data['phase'],dtype=int)
            raw_samples = np.array(data['samples'],dtype=int)
            del data

            #load pedestal waveforms
            Ped_infile = "/baseline_8_block_sync_avg_1556_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            data = np.load(Ped_dir+Ped_infile)
            ped_block = np.array(data['block'],dtype=int)
            ped_phase = np.array(data['phase'],dtype=int)
            ped_samples = np.array(data['baseline'],dtype=int)  
            del data

            #loop through each block
            samp_size = raw_samples.shape
            samples = np.zeros(samp_size)
            phase_vals = np.unique(phase)
            block_vals = np.unique(block)
            b_length = len(block_vals)
            '''
	    for q, b_val in enumerate(block_vals):
                
		#if((q+1)%8==0):
		sys.stdout.write('\r')
		sys.stdout.write("[%-64s] %d%% Block %s of %s" % ('='*int((q+1)*64./float(b_length)) , (q)*100./float(b_length), b_val+1, b_length))
		sys.stdout.flush()

                #loop through each phase
		for p_val in phase_vals:
                    base_index = b_val*4+p_val
                    baseline = ped_samples[base_index]
		    #loop through each event
		    for i in range(samples.shape[0]):
			if block[i]==b_val:
                            if phase[i]==p_val:
                                samples[i,:] = raw_samples[i]-baseline
	    '''
            for i in range(samples.shape[0]):
                #base_index = block[i]*nPhases+(phase[i]-4)/8
                base_index = block[i]*nPhases+(phase[i])/8
                #base_index = block[i]*nPhases+phase[i]
                baseline = ped_samples[base_index]
                samples[i,:] = raw_samples[i]-baseline		    
                  
            #save calibrated samples to compressed numpy array
            #To load data, use np.load('path/filename.npz')['keyword'] 
            #List of keywords: run, asic, channel, event, block, phase, samples
            #EXAMPLE: blockphase = np.load('path/filename.npz')['phase']
            np.savez_compressed(outfile,run=run,asic=asic,channel=channel,event=event,block=block,phase=phase,samples=samples)

        #sys.stdout.write('\n')

print 'Job Time: ',time.time()-start_time
           
