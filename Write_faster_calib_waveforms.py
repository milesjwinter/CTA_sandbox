"""
Miles Winter
Run raw waveforms through transfer function then write out to text file
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

start_time = time.time()

#Run information
runs = np.arange(314490,314500,1)
runs = np.append(runs,np.arange(314513,314533,1))
Ncells = 8*32

#specify transfer function and data directory
TF_dir = "Eight_Block_TF"
Data_dir = "/Users/mileswinter/test_suite/runFiles"
#Loop through each run
for Nrun in runs:
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):
            #create output file for calibrated samples
            outfile = "calib_waveforms/calib_samples_Run"+str(Nrun)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            
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

            #Convert samples from ADC counts to mV w/ transfer function
            samp_size = raw_samples.shape
            samples = np.zeros(samp_size)
            phase_vals = np.unique(phase)
            block_vals = np.unique(block)
            b_length = len(block_vals)
            #loop through each block
	    for q, b_val in enumerate(block_vals):
                
		#if((q+1)%8==0):
		sys.stdout.write('\r')
		sys.stdout.write("[%-64s] %d%% Block %s of %s" % ('='*int((q+1)*64./float(b_length)) , (q)*100./float(b_length), b_val+1, b_length))
		sys.stdout.flush()

                #loop through each phase
		for p_val in phase_vals:
		    #Load transfer function for the correct block and phase
                    TF_infile = "/TF_table_ASIC"+str(Nasic)+"CH"+str(Nchannel)+"BN"+str(int(b_val))+"BP"+str(int(p_val))+".npz"
                    data = np.load(TF_dir+TF_infile)['arr_0']
                    ADC_counts = np.array(data[:,0],dtype=int)
                    ADC_to_mV = np.array(data[:,1:])
                    del data
		    for i in range(samples.shape[0]):
			if block[i]==b_val:
                            if phase[i]==p_val:
                                for j in range(samp_size[1]):
                                    ADC_index = np.where(ADC_counts==raw_samples[i,j])
                                    samples[i,j] = ADC_to_mV[ADC_index[0],j]
			    
                  
            #save calibrated samples to compressed numpy array
            #To load data, use np.load('path/filename.npz')['keyword'] 
            #List of keywords: run, asic, channel, event, block, phase, samples
            #EXAMPLE: blockphase = np.load('path/filename.npz')['phase']
            np.savez_compressed(outfile,run=run,asic=asic,channel=channel,event=event,block=block,phase=phase,samples=samples)

        sys.stdout.write('\n')

print 'Job Time: ',time.time()-start_time
           
