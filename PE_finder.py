"""
Miles Winter
Pulse Integration
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

#Run information
runs = np.arange(314513,314533,1)
base_run = [314510]
runs = np.append(runs,np.arange(314490,314500,1))
vpeds = [1106.]

#Declare constants
Npeaks = 10       #Number of peaks to search for in each event
l_edge = 8       #num of - bins added to signal integration window
r_edge = 8       #num of + bins added to signal integration window
Lp = 2.5         #lower percentile (2sigma currently)
Up = 98.0 #99.85, 97.5        #upper percentile (3sigma currently)
Ncells = 256

total_pulses = np.array([])
total_max = np.array([])

#Loop through each run
for run_num in runs:
    #Loop through each asic
    for asic in range(1):
        #Loop through each channel
        for channel in range(14,15): 
            #Load samples
            data = np.loadtxt("/Users/mileswinter/test_suite/runFiles/sampleFileLarge_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            ievt = np.array(data[:,2]) 
            blocknum = np.array(data[:,3]) 
            blockphase = np.array(data[:,4])
            samples = np.array(data[:,5:],dtype=int)

            #initialize a few variables and temp arrays
            Nevents = len(samples[:,0])                #num of events in samples
            pulse_area = np.zeros((Nevents,Npeaks))   #array to hold integrated pulses
            pulse_max = np.zeros((Nevents,Npeaks))    #array to hold max position
            baseline = 0.                              #clear baseline for each loop
            
            #Load baseline
            data = np.loadtxt("/Users/mileswinter/test_suite/runFiles/sampleFileLarge_run"+str(base_run[0])+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            base_samples = np.array(data[:,5:],dtype=int)
            baseline = np.percentile(base_samples,50.,axis=0)
            ''' 
            data = np.loadtxt("run_files/flasher_av_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            for i in range(1,len(data[:,0])):
                if data[i,4]>0.0:
                    baseline = np.array(data[i,4:], dtype=int)
            '''
            #Load transfer function
            data = np.loadtxt("TF_Tables/TF_table_runs_314465_314487.txt")
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
            
            #Correct samples with baseline subtraction in mV.
            new_baseline = np.percentile(mV_samples[:,:64],50.,axis=1)
            new_baseline = new_baseline.reshape(len(new_baseline),1)
            #corr_samples = mV_samples-mV_baseline 
            #corr_samples = mV_samples-new_baseline
            corr_samples = mV_samples-np.percentile(mV_samples,50,axis=0)
            low_lim, up_lim = np.percentile(corr_samples, [Lp,Up])
            max_vals = np.amax(corr_samples, axis=1)
            min_vals = np.amin(corr_samples, axis=1)
            temp_ped = 0.

            #Loop through all events in samples
            for i in range(Nevents):
            #for i in range(5):
                #exclude outliers and no pulse events then fill modified temp sample array
                temp_samples = []            #modified samples array for peak finding
                signal_begin = [0]*Npeaks       #start of signal integration window 
                signal_end = [Ncells+1]*Npeaks   #end of signal integration window (arbitrary initial value)

                if max_vals[i] >= np.mean(up_lim): # and min_vals[i] >= np.mean(low_lim):
                    for u in range(Ncells):
                        #if above noise cutoff, keep exact value
                        if corr_samples[i,u] >= np.mean(up_lim):
                            temp_samples.append(corr_samples[i,u])
                        #if below noise cutoff, set to average of the baseline
                        else:
                            temp_samples.append(temp_ped)
               
                    #search for signal start
                    for j in range(Npeaks):
                        begin_array = []
                        if j == 0:
                            for m in range(len(temp_samples)):
                                if temp_samples[m] > temp_ped:
                                    begin_array.append(m)
                        else:
                            for m in range(signal_end[j-1], len(temp_samples)): 
                                if temp_samples[m] > temp_ped:
                                    begin_array.append(m)
                        if begin_array != []:
                            signal_begin[j] = begin_array[0]

                            #search for signal end
                            end_array = temp_samples[signal_begin[j]:]
                            if temp_ped in end_array:
                                if len(temp_samples) - end_array.index(temp_ped) - signal_begin[j] >= r_edge:
                                    signal_end[j] = end_array.index(temp_ped) + signal_begin[j] + r_edge
                                else:
                                    signal_end[j] = len(temp_samples) #end_array.index(temp_ped) + signal_begin[j]
                            else:
                                signal_end[j] = len(temp_samples)+1
                            '''
                            #find max position
                            if signal_end[j] < 64:
                                 pulse_max[i,j] = max(temp_samples[signal_begin[j]:signal_end[j]])
                            else: 
                                 pulse_max[i,j] = max(temp_samples[signal_begin[j]:])
                            '''

                            #perform signal integration for all samples (0-64)
                            if signal_begin[j] <= l_edge:
                                if signal_begin[j] == 0:
                                    pulse_area[i,j] = 0.
                                else:
                                    pulse_area[i,j] = sum(corr_samples[i,:signal_end[j]+r_edge])
                            elif signal_end[j] >= len(temp_samples) - r_edge:
                                if signal_end[j] > len(temp_samples):
                                    pulse_area[i,j] = 0
                                else:
                                    pulse_area[i,j] = sum(corr_samples[i,signal_begin[j]-l_edge:])
                            else:
                                pulse_area[i,j] = sum(corr_samples[i,signal_begin[j]-l_edge:signal_end[j]+r_edge])
                    
                    ''' 
                    print pulse_area[i,:]           
                    plt.figure()
                    plt.plot(np.arange(0,64,1),corr_samples[i,:])
                    plt.plot(np.arange(0,64,1),temp_samples)
                    for w in range(3):
                        plt.axvline(signal_begin[w]-l_edge,color='k', lw=1.5, linestyle='dashed')
                        plt.axvline(signal_end[w],color='k', lw=1.5, linestyle='dashed')
                    plt.xlabel('times (ns)')
                    plt.ylabel('ADC Counts')
                    plt.xlim([0,64])
                    plt.ylim([-20,150])
                    plt.title('Charge = %s' % pulse_area[i,:])
                    plt.show()
                    '''
            print 'Run: ',run_num, ' job time: ', time.time() - start_time
            
            for p in range(Nevents):
                #total_pulses = np.append(total_pulses,np.amax(pulse_area[p,:]))
                #total_max = np.append(total_max,np.amax(pulse_area[p,:]))
                
                for q in range(Npeaks):
                    if pulse_area[p,q] > 0:
                        total_pulses = np.append(total_pulses,pulse_area[p,q])
                    if pulse_max[p,q] > 0:
                        total_max = np.append(total_max,pulse_max[p,q])
                

plt.figure(1,figsize=(20,10))
plt.hist(total_pulses[total_pulses>0],bins=np.arange(0,1000,10))
#plt.hist(total_pulses[total_pulses>0]/75.,bins=np.arange(0,8,.1))
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.show()

'''            
            plt.figure(2)
            plt.hist(total_max,bins=Ncells)
            plt.xlabel('Pulse Maximum (ns)')
            plt.ylabel('counts')
            plt.xlim([0,Ncells])
            plt.show()
            #Store results in a text file
            #np.savetxt("pulse_files/pulse_int_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt", pulse_area)
'''



