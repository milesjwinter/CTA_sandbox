"""
Miles Winter
Pulse Integration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import sys

start_time = time.time()

#Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2.*sigma**2))

#Run information
#runs = np.arange(319428,319435,1)
#runs = np.arange(319518,319558,1)
#runs = np.arange(319408,319418,1)
#runs = np.append(runs,np.arange(319436,319446,1))
#runs = [319247]
#runs = np.arange(319252,319272,1)
#runs = np.arange(319208,319214,1)
#runs = np.arange(319147,319162,1)
#runs = np.arange(318745,318765,1)
#runs = np.arange(318775,318782,1)
#runs = np.arange(316170,316180,1)
#runs = np.arange(317520, 317536,1)
#runs = np.arange(314515,314533,1)
#runs = [316170,316171,316172,316173,316174,316175,316176,316177,316178,316179]
#runs = [316172]
#runs = np.arange(314490,314500,1)
#runs = np.append(runs,np.arange(314513,314533,1))
#runs = [314530, 314531, 314532]
#runs = np.arange(320235,320244,1)
#runs = np.append(runs,np.arange(320245,320247,1))
runs = np.arange(320247,320251,1)
Ncells = 8*32
window_start = 180#35#150
window_stop = 200#55#156
low_window_size = 4#12
up_window_size = 4#12
fixed_position = 153#44
fit_size = 10
nPhases = 4


#load and calculate baseline waveform
#data = np.load("run_files/sampleFileAllBlocks_run317514ASIC0CH14.npz")
data = np.load("run_files/sampleFileAllBlocks_run316151ASIC0CH14.npz")
base_samples = data['samples']
baseline = np.mean(base_samples,axis=0)
#corr_baseline = baseline-np.mean(base_samples[:,:64])
#baseline = baseline.reshape(len(baseline),1)
del data
'''
data = np.load("run_files/sampleFileAllBlocks_run316179ASIC0CH14.npz")
shift_samples = data['samples']
new_ped = shift_samples - baseline 
shift_val = np.mean(new_ped[:,:64])
'''
#Loop through each run
for run_num in runs:
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15): 
           
            #Load samples
            #data = np.load("run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
            #data = np.load("calib_waveforms/calib_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            data = np.load("base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            run = data['run']
            asic = data['asic']
            channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            #samp_avg = np.mean(samples[:,:24])*np.ones(samples.shape)
            #alt_ped = shift_val*np.ones(samples.shape)
            del data
            
            #initialize a few variables and temp arrays
            Nevents = len(event)         #num of events in samples
            charge = np.zeros(Nevents)
            position = np.zeros(Nevents,dtype=int)
            amplitude = np.zeros(Nevents)
            ped_max = np.zeros(Nevents)
            ped_min = np.zeros(Nevents)
            fit_charge = np.zeros(Nevents,dtype=int)
            fit_position = np.zeros(Nevents,dtype=int)
            fit_amplitude = np.zeros(Nevents)
            fit_width = np.zeros(Nevents)
       
            #Correct samples with pedestal subtraction in mV.
            #pedestal = np.mean(samples[:,:24],axis=1)
            pedestal = np.mean(samples[:,:128],axis=1)
            pedestal = pedestal.reshape(len(pedestal),1)
            #adjusted_samples = samples-baseline
            #corr_samples = samples-samp_avg-corr_baseline
            #corr_samples = samples-baseline
            #corr_samples = samples-baseline-alt_ped
            corr_samples = samples-pedestal
            #corr_samples = samples-np.mean(samples,axis=0)
            #pedestal = np.mean(adjusted_samples[:,:32],axis=1)
            #pedestal = pedestal.reshape(len(pedestal),1)
            #corr_samples = adjusted_samples-pedestal
            #corr_samples = adjusted_samples
            '''
            #load pedestal waveforms
            Ped_infile = "/baseline_TF_avg_"+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            data = np.load(Ped_dir+Ped_infile)
            ped_block = np.array(data['block'],dtype=int)
            ped_phase = np.array(data['phase'],dtype=int)
            ped_samples = np.array(data['baseline'],dtype=int)
            del data

            corr_samples = np.zeros(samples.shape)
            for i in range(samples.shape[0]):
                base_index = block[i]*nPhases+phase[i]
                baseline = ped_samples[base_index]
                corr_samples[i,:] = samples[i]-baseline

            '''
            #Loop through all events in samples
            print "Generating Charge Spectrum: Asic ",Nasic," Channel ", Nchannel
            for i in range(Nevents): 
                if(i%100==0):
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-100s] %.1f%%" % ('='*int((i)*100./(Nevents)) , (i)*100./(Nevents)))
                    sys.stdout.flush()
                    
                #determine pulse position
                position[i] = window_start+np.argmax(corr_samples[i,window_start:window_stop])
                amplitude[i] = corr_samples[i,position[i]]
                start_integrate = position[i]-low_window_size
                stop_integrate = position[i]+up_window_size
                if start_integrate<0:
                    start_integrate = 0
                if stop_integrate > Ncells:
                    stop_integrate = Ncells
                 
                #start_integrate = fixed_position-low_window_size
                #stop_integrate = fixed_position+up_window_size
                charge[i] = np.sum(corr_samples[i,start_integrate:stop_integrate])
                ped_max[i] = np.amax(corr_samples[i,:128])
                ped_min[i] = np.amin(corr_samples[i,:128])
            
                '''
                if amplitude[i]>5.:                     
                    try:                
                        fit_start =  position[i]-fit_size
                        fit_stop =  position[i]+fit_size
                        x = np.arange(fit_start,fit_stop,1)
                        y = corr_samples[i,fit_start:fit_stop]
                        #estimate mean and standard deviation
                        mean = np.mean(y)
                        sigma = np.std(y)
                        #do the fit!
                        popt, pcov = curve_fit(gauss_function, x, y,bounds=([0,140,0],[np.inf,170,8]) )
                        #popt, pcov = curve_fit(gauss_function, x, y, p0 = [amplitude[i], position[i], sigma])
                        points = np.arange(0,Ncells,1)
                        #start_int = int(popt[1]-2.*popt[2])
                        #stop_int = int(popt[1]+2.*popt[2])
                        start_int = int(popt[1]-window_size)
                        stop_int = int(popt[1]+window_size)
                        fit_charge[i] = np.sum(corr_samples[i,start_int:stop_int])
                        #charge[i] = np.sqrt(2.*np.pi)*popt[0]*popt[2]
                        fit_amplitude[i] = popt[0]
                        fit_position[i] = popt[1]
                        fit_width[i] = popt[2]

                        if position[i] == 153:   
			    #plot the fit results
			    plt.figure()
			    plt.plot(points,gauss_function(points, a=popt[0],x0=popt[1],sigma=popt[2]))
			    plt.plot(points,corr_samples[i,:],'k')
			    plt.axvline(popt[1],color='r', lw=1.5, linestyle='dashed')
                            plt.axvline(start_int,color='k', lw=1.5, linestyle='dashed')
                            plt.axvline(stop_int,color='k', lw=1.5, linestyle='dashed')
			    #plt.axvline(int(popt[1]-2.*popt[2]),color='k', lw=1.5, linestyle='dashed')
			    #plt.axvline(int(popt[1]+2.*popt[2]),color='k', lw=1.5, linestyle='dashed')
			    plt.xlabel('times (ns)')
			    plt.ylabel('ADC Counts')
			    plt.xlim([0,Ncells])
			    plt.ylim([-20,150])
			    plt.title('Fit Params: $A$='+str(np.round(popt[0],1))+', $\mu$='+str(np.round(popt[1],1))+', $\sigma$='+str(np.round(popt[2],1)))
                            plot_label = "plots/fit_charge_spec/fitted_sample_pulse"+str(i)+".png"
                            plt.savefig(plot_label)
			    plt.show()
                        
                    except RuntimeError:
                        fit_charge[i]=0
                else:
                    fit_charge[i] = np.sum(corr_samples[i,start_integrate:stop_integrate])                        
                '''
                   
	    sys.stdout.write('\n')
	    print 'Run: ',run_num, ' job time: ', time.time() - start_time
	    
	    #Store results in a text file
	    print "Saving charge spectrum for run ",run_num," to external file"
	    #outfile = "charge_spectrum_files/simple_alt_fit_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            outfile = "charge_spectrum_files/ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
	    np.savez_compressed(outfile, run=run, asic=asic, channel=channel, event=event, block=block, phase=phase, charge=charge, position=position, amplitude=amplitude, ped_max=ped_max, ped_min=ped_min)
            #np.savez_compressed(outfile, run=run, asic=asic, channel=channel, event=event, block=block, phase=phase, charge=charge, position=position, amplitude=amplitude, fit_width=fit_width,fit_charge=fit_charge, fit_position=fit_position, fit_amplitude=fit_amplitude)
	    print 'Analysis complete: job time ', time.time() - start_time

	'''
	#plot the fit results
	plt.figure()
	plt.plot(points,gauss_function(points, a=popt[0],x0=popt[1],sigma=popt[2]))
	plt.plot(points,corr_samples[i,:],'k')
i	plt.axvline(popt[1],color='r', lw=1.5, linestyle='dashed')
	plt.axvline(int(popt[1]-2.*popt[2]),color='k', lw=1.5, linestyle='dashed')
	plt.axvline(int(popt[1]+2.*popt[2]),color='k', lw=1.5, linestyle='dashed')
	plt.xlabel('times (ns)')
	plt.ylabel('ADC Counts')
	plt.xlim([0,Ncells])
	plt.ylim([-20,150])
	plt.title('Charge = %s' % charge[i])
	plt.show()
	'''

	'''
	plt.figure()
	plt.plot(np.arange(0,Ncells,1),corr_samples[i,:])
	plt.axvline(position[i],color='r', lw=1.5, linestyle='dashed')
	plt.axvline(start_integrate,color='k', lw=1.5, linestyle='dashed')
	plt.axvline(stop_integrate,color='k', lw=1.5, linestyle='dashed')
	plt.xlabel('times (ns)')
	plt.ylabel('ADC Counts')
	plt.xlim([0,Ncells])
	plt.ylim([-20,150])
	plt.title('Charge = %s' % charge[i])
	plt.show()
	'''
