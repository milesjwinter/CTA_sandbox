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
runs = np.arange(316170,316180,1)
#runs = [316170,316171,316172,316173,316174,316175,316176,316177,316178,316179]
#runs = [316172]

Ncells = 8*32
window_start = 145
window_stop = 160
window_size = 8

#Loop through each run
for run_num in runs:
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15): 
           
            #Load samples
            data = np.load("calib_waveforms/calib_ped_sub_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
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
            charge = np.zeros(Nevents)
            position = np.zeros(Nevents,dtype=int)
            amplitude = np.zeros(Nevents)
            ped_max = np.zeros(Nevents)
       
            #Correct samples with pedestal subtraction in mV.
            pedestal = np.mean(samples[:,96:128],axis=1)
            pedestal = pedestal.reshape(len(pedestal),1)
            corr_samples = samples-pedestal
     
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
                start_integrate = 153-4
                stop_integrate = 153+8
                charge[i] = np.sum(corr_samples[i,start_integrate:stop_integrate])
                ped_max[i] = np.amax(corr_samples[i,96:128])

                '''                 
                try:                
                    if amplitude[i]>10.:
                        x = np.arange(start_integrate,stop_integrate,1)
                        y = corr_samples[i,start_integrate:stop_integrate]
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
                        charge[i] = np.sum(corr_samples[i,start_int:stop_int])
                        #charge[i] = np.sqrt(2.*np.pi)*popt[0]*popt[2]
                        amplitude[i] = popt[0]
                        position[i] = popt[1]
                        width[i] = popt[2]

                               
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
                    charge[i]=0
                '''
	    sys.stdout.write('\n')
	    print 'Run: ',run_num, ' job time: ', time.time() - start_time
	    
	    #Store results in a text file
	    print "Saving charge spectrum for run ",run_num," to external file"
	    #outfile = "charge_spectrum_files/simple_fit_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            outfile = "charge_spectrum_files/simple_ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
	    np.savez_compressed(outfile, run=run, asic=asic, channel=channel, event=event, block=block, phase=phase, charge=charge, position=position, amplitude=amplitude, ped_max=ped_max)
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
