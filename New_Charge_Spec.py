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
runs = np.arange(316170,316180,1)
#runs = [316084]

#Declare constants
Npeaks = 4       #Number of peaks to search for in each event
l_edge = 8       #num of - bins added to signal integration window
r_edge = 10       #num of + bins added to signal integration window
min_space = 8 #2    #minimum number of bins between two peaks
Lp = 2.5         #lower percentile (2sigma currently)
Up = 99.85 #99.85, 97.5        #upper percentile (3sigma currently)

#Loop through each run
for run_num in runs:
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15): 
           
            #Load samples
            data = np.load("calib_waveforms/calib_ped_sub_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            Nevent = data['event']
            Nblock = data['block']
            Nphase = data['phase']
            samples = data['samples']
            del data
            
            #initialize a few variables and temp arrays
            Nevents = len(Nevent)         #num of events in samples
            Ncells = len(samples[0,:])
            run = np.array([])
            asic = np.array([])
            channel = np.array([])
            event  = np.array([])
            block = np.array([])
            phase = np.array([])
            charge = np.array([])
            max_pos = np.array([])
            max_val = np.array([])     
       
            #Correct samples with baseline subtraction in mV.
            pedestal = np.percentile(samples[:,:128],50.,axis=1)
            pedestal = pedestal.reshape(len(pedestal),1)
            corr_samples = samples-pedestal
     
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
                pulse_area = np.zeros(Npeaks)   #array to hold integrated pulses
                pulse_max = np.zeros(Npeaks)    #array to hold max position

                if max_vals[i] >= np.mean(up_lim): 
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
                            
                            #find max position
                            if signal_end[j] < Ncells:
                                 pulse_max[j] = np.argmax(temp_samples[signal_begin[j]:signal_end[j]])+signal_begin[j]
                            else: 
                                 pulse_max[j] = np.argmax(temp_samples[signal_begin[j]:])+signal_begin[j]
                            

                            #perform signal integration for all samples
                            if j==0:
                                if signal_begin[j] <= l_edge:
                                    if signal_begin[j] == 0:
                                        pulse_area[j] = 0.
                                    else:
                                        pulse_area[j] = sum(corr_samples[i,:signal_end[j]+min_space])
                                elif signal_end[j] >= len(temp_samples) - r_edge:
                                    if signal_end[j] > len(temp_samples):
                                        pulse_area[j] = 0
                                    else:
                                        pulse_area[j] = sum(corr_samples[i,signal_begin[j]-min_space:])
                                else:
                                    pulse_area[j] = sum(corr_samples[i,signal_begin[j]-min_space:signal_end[j]+min_space])
                            else:
                                if pulse_max[j]-pulse_max[j-1]<r_edge:
                                    pulse_area[j-1]=0.
                                    if signal_begin[j-1] <= l_edge:
                                        pulse_area[j] = sum(corr_samples[i,:signal_end[j]+min_space])
                                    elif signal_end[j] >= len(temp_samples) - r_edge:
                                        if signal_end[j] > len(temp_samples):
                                            pulse_area[j] = 0
                                        else:
                                            pulse_area[j] = sum(corr_samples[i,signal_begin[j-1]-min_space:])
                                    else:
                                        pulse_area[j] = sum(corr_samples[i,signal_begin[j-1]-min_space:signal_end[j]+min_space])
                                else:
                                    if signal_begin[j] <= l_edge:
                                        if signal_begin[j] == 0:
                                            pulse_area[j] = 0.
                                        else:
                                            pulse_area[j] = sum(corr_samples[i,:signal_end[j]+min_space])
                                    elif signal_end[j] >= len(temp_samples) - r_edge:
                                        if signal_end[j] > len(temp_samples):
                                            pulse_area[j] = 0
                                        else:
                                            pulse_area[j] = sum(corr_samples[i,signal_begin[j]-min_space:])
                                    else:
                                        pulse_area[j] = sum(corr_samples[i,signal_begin[j]-min_space:signal_end[j]+min_space])
                    
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
                    plt.title('Charge = %s' % pulse_area[:])
                    plt.show()
                    '''
                    for w in range(Npeaks):
                        if pulse_area[w]>0:
                            run = np.append(run,run_num)
                            asic = np.append(asic,Nasic)
                            channel = np.append(channel,Nchannel)
                            event = np.append(event,Nevent[i])
                            block = np.append(block,Nblock[i])
                            phase = np.append(phase,Nphase[i])
                            charge = np.append(charge,pulse_area[w])
                            max_pos = np.append(max_pos,pulse_max[w])
                            max_val = np.append(max_val,corr_samples[i,int(pulse_max[w])]) 
                            
            sys.stdout.write('\n')
            print 'Run: ',run_num, ' job time: ', time.time() - start_time
            
            #Store results in a text file
            print "Saving charge spectrum for run ",run_num," to external file"
            outfile = "charge_spectrum_files/charge_alt_ped_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            np.savez_compressed(outfile, run=run, asic=asic, channel=channel, event=event, block=block, phase=phase, charge=charge, max_pos=max_pos, max_val=max_val)
print 'Analysis complete: job time ', time.time() - start_time


