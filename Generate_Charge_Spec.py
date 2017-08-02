"""
Miles Winter
Pulse Integration
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

start_time = time.time()

#Run information
#runs = np.arange(314513,314533,1)
#runs = [316175,316176,316177,316178,316179]
runs = [316175]
vpeds = [1106.]

#Declare constants
Npeaks = 6       #Number of peaks to search for in each event
l_edge = 10       #num of - bins added to signal integration window
r_edge = 10       #num of + bins added to signal integration window
Lp = 2.5         #lower percentile (2sigma currently)
Up = 99.9 #99.85, 97.5        #upper percentile (3sigma currently)
Ncells = 128

total_pulses = np.array([])
total_pulse_count = np.array([])
total_max = np.array([])

#Loop through each run
for run_num in runs:
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15): 
            #Load samples
            infile = "run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            data = np.load(infile)

            #run = data['run']
            #asic = data['asic']
            #channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            del data

            #initialize a few variables and temp arrays
            Nevents = len(samples[:,0])                #num of events in samples
            pulse_area = np.zeros((Nevents,Npeaks))   #array to hold integrated pulses
            pulse_max = np.zeros((Nevents,Npeaks))    #array to hold max position
                       
            #Correct samples with baseline subtraction in mV.
            #pedestal = np.percentile(samples[:,:128],50.,axis=1)
            #pedestal = pedestal.reshape(len(pedestal),1)
            #corr_samples = samples-pedestal
            corr_samples = samples[:,:Ncells]-np.mean(samples[:,:Ncells],axis=0)
     
            low_lim, up_lim = np.percentile(corr_samples[:,:128], [Lp,Up])
            max_vals = np.amax(corr_samples, axis=1)
            min_vals = np.amin(corr_samples, axis=1)
            temp_ped = 0.

            #Loop through all events in samples
            print "Generating Charge Spectrum: Asic ",Nasic," Channel ", Nchannel
            for i in range(Nevents):
                if(i%100==0):
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-100s] %.1f%%" % ('='*int((i)*100./(Nevents)) , (i)*100./(Nevents)))
                    sys.stdout.flush()

                #exclude outliers and no pulse events then fill modified temp sample array
                temp_samples = []            #modified samples array for peak finding
                signal_begin = [0]*Npeaks       #start of signal integration window 
                signal_end = [Ncells+1]*Npeaks   #end of signal integration window (arbitrary initial value)

                if max_vals[i] >= 8.41: #np.mean(up_lim): 
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
                    plt.figure()
                    plt.plot(np.arange(0,Ncells,1),corr_samples[i,:])
                    plt.plot(np.arange(0,Ncells,1),temp_samples)
                    for w in range(Npeaks):
                        plt.axvline(signal_begin[w]-l_edge,color='k', lw=1.5, linestyle='dashed')
                        plt.axvline(signal_end[w],color='k', lw=1.5, linestyle='dashed')
                    plt.xlabel('times (ns)')
                    plt.ylabel('ADC Counts')
                    plt.xlim([0,Ncells])
                    plt.ylim([-20,150])
                    plt.title('Charge = %s' % pulse_area[i,:])
                    plt.show()
                    '''
            sys.stdout.write('\n')
            print 'Run: ',run_num, ' job time: ', time.time() - start_time
            
            for p in range(Nevents):
                total_pulses = np.append(total_pulses,np.amax(pulse_area[p,:]))
                total_max = np.append(total_max,np.amax(pulse_area[p,:]))
                total_pulse_count = np.append(total_pulse_count,np.sum(pulse_area[p,:]>0))
                '''
                for q in range(Npeaks):
                    if pulse_area[p,q] > 0:
                        total_pulses = np.append(total_pulses,pulse_area[p,q])
                    if pulse_max[p,q] > 0:
                        total_max = np.append(total_max,pulse_max[p,q])
                '''

#Store results in a text file
#print "Saving charge spectrum to external file"
#np.savez_compressed("charge_spectrum_files/charge_spectrum_run314515_ASIC0CH14.npz", total_pulses)
'''
plt.figure(1,figsize=(20,10))
plt.hist(total_pulses[total_pulses>0],bins=np.arange(0,1000,5))
#plt.hist(total_pulses[total_pulses>0]/75.,bins=np.arange(0,8,.1))
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.show()
'''
print 'Analysis complete: job time ', time.time() - start_time
plt.figure()
plt.hist(total_pulse_count,bins=np.arange(0,7,1))
plt.xlabel('Dark Pulses Per Waveform',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.show()

