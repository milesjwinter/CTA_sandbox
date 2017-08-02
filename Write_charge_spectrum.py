import numpy as np
import matplotlib.pyplot as plt
import time
import sys

start_time = time.time()

#run parameters and temp arrays
runs = np.arange(316170,316180,1)
#runs = [316170]


#Loop through each run
for run_num in runs:
    print "Loading charge spectrum: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load charge spectrum
            data = np.load("charge_spectrum_files/simple_ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            run = data['run']
            asic = data['asic']
            channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            charge = data['charge']
            position = data['position']
            amplitude = data['amplitude']
            width = data['width']
            del data

            Nevents = len(event)
            total_charge = np.array([])
            total_position = np.array([])
            total_amplitude = np.array([])
            total_width = np.array([])
            print "Applying data cuts: Asic ",Nasic," Channel ", Nchannel
	    for i in range(Nevents):
                if(i%100==0):
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-100s] %.1f%%" % ('='*int((i)*100./(Nevents)) , (i)*100./(Nevents)))
                    sys.stdout.flush()
                if charge[i]>0:
	            if amplitude[i]>=0:
                    #if width[i]<=10.:
		        if position[i]>=152 and position[i]<=153:
		        #if position[i]==153:
                            total_charge = np.append(total_charge,charge[i])
                            #total_charge = np.append(total_charge,np.sqrt(2.*np.pi)*amplitude[i]*np.random.normal(5, .9))
                            #total_position = np.append(total_position,position[i])
                            #total_amplitude = np.append(total_amplitude,amplitude[i])
                            #total_width = np.append(total_width,width[i])
            
            charge_hist, charge_bins = np.histogram(total_charge,bins=np.arange(0,1201,1))
            outfile = "charge_spectrum_files/stored_charge_histogram_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"     
            np.savez_compressed(outfile,charge_hist=charge_hist)      
            sys.stdout.write('\n')

print 'Analysis complete: job time ', time.time() - start_time
'''
#Plot histogram
plt.figure(1,figsize=(16,8))
plt.hist(total_charge,bins=np.arange(0,1000,4))
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()

plt.figure(2,figsize=(16,8))
plt.hist(total_position,bins=np.arange(148,158,1))
plt.title('Position of Max Pulse Height',fontsize=24)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()

plt.figure(3,figsize=(16,8))
plt.hist(total_amplitude,bins=np.arange(0,101,0.5))
plt.title('Maximum Pulse Amplitude',fontsize=24)
plt.xlabel('ADC Counts (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()
'''
'''
plt.figure(4,figsize=(16,8))
plt.plot(total_amplitude,total_width,'ok',alpha=0.1)
plt.title('Pulse Width vs Amplitude',fontsize=24)
plt.xlabel('Amplitude (mV)',fontsize=20)
plt.ylabel('Width (ns)',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()
'''

