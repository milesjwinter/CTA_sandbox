"""
Miles Winter
Testing Transfer Function Results
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

#Run information
#runs = np.arange(313345,313365,1) #[313352] #[312070]
#runs = [313348,313349,313350,313351,313352,313353,313369]
runs = [313369]
Ncells = 64

mV_base_runs = np.zeros((len(runs),Ncells))
#Loop through each run
for run_num in runs:
    print run_num
    #Loop through each asic
    for asic in range(1):
        #Loop through each channel
        for channel in range(1):
            #Load samples
            data = np.loadtxt("run_files/sampleFileLarge_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            ievt = np.array(data[:,2])
            blocknum = np.array(data[:,3])
            blockphase = np.array(data[:,4])
            samples = np.array(data[:,5:],dtype=int)

            #initialize a few variables and temp arrays
            Nevents = len(samples[:,0])                #num of events in samples
            baseline = 0.                              #clear baseline for each loop

            #Load baseline
            data = np.loadtxt("run_files/flasher_av_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            for i in range(1,len(data[:,0])):
                if data[i,4]>0.0:
                    baseline = np.array(data[i,4:], dtype=int)

            #Load transfer function
            data = np.loadtxt("transfer_function_table.txt")
            ADC_counts = np.array(data[:,0],dtype=int)
            ADC_to_mV = np.array(data[:,1:])
            
            #Convert samples from ADC counts to mV w/ transfer function
            samp_size = samples.shape
            mV_samples = np.zeros(samp_size)
           
            for i in range(samp_size[0]):
                for j in range(samp_size[1]):
                    ADC_index = np.where(ADC_counts==samples[i,j])
                    mV_samples[i,j] = ADC_to_mV[ADC_index,j]
            
            mV_baseline=np.zeros(Ncells)
            for p in range(len(baseline)):
                ind = np.where(ADC_counts==baseline[p])
                mV_baseline[p] = ADC_to_mV[ind,p]
            if run_num==313369:
                mV_base_runs[-1,:]=mV_baseline
            else:
                mV_base_runs[run_num-runs[0],:]=mV_baseline
            #Correct samples with baseline subtraction in mV.
            fast_baseline = np.percentile(mV_samples,50.,axis=0)
            corr_mV_samples = mV_samples-fast_baseline
            
            #find max position in samples
            mV_max_pos = np.argmax(mV_samples,axis=1)
            mV_min_pos = np.argmin(mV_samples,axis=1)
            mV_max_val = np.amax(mV_samples,axis=1)
            mV_min_val = np.amin(mV_samples,axis=1)

            #calculate std 
            std_cell = np.std(mV_samples,axis=0)
            std_samp = np.std(mV_samples,axis=1)
            '''           
            plt.figure(1)
            for q in range(5000,5010):
                plt.plot(np.arange(0,Ncells,1),mV_samples[q,:])
            plt.show()
            
            plt.figure(2)
            plt.hist(std_cell,bins=50)
            plt.title('Std Cells')
            plt.xlabel('$1\sigma$ Std (mV)')
            plt.ylabel('counts')
            plt.show()   

            plt.figure(3)
            plt.hist(std_samp,bins=50)
            plt.title('Std Samples')
            plt.xlabel('$1\sigma$ Std (mV)')
            plt.ylabel('counts')
            plt.show()
            '''

            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].hist(mV_max_pos,bins=np.arange(0,Ncells,1))
            axarr[0].set_title('Max Position')
            axarr[0].set_ylabel('Counts')
            axarr[1].hist(mV_min_pos,bins=np.arange(0,Ncells,1))
            axarr[1].set_title('Min Position')
            axarr[1].set_ylabel('Counts')
            axarr[1].set_xlabel('time (ns)')
            plt.show()

            f, axar = plt.subplots(2, sharex=True)
            axar[0].hist(mV_max_val,bins=100)
            axar[0].set_title('Max Value')
            axar[0].set_ylabel('Counts')
            axar[1].hist(mV_min_val,bins=100)
            axar[1].set_title('Min Value')
            axar[1].set_ylabel('Counts')
            axar[1].set_xlabel('mV')
            plt.show()
            
'''
plt.figure(5)
for i in range(len(runs)):
    plt.plot(np.arange(0,Ncells,1),mV_base_runs[i,:])
plt.xlabel('times (ns)')
plt.ylabel('Counts (mV)')
plt.show()         
'''
