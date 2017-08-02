import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
import time
import sys

start_time = time.time()

#Run information
#runs = np.arange(316170,316180,1)
runs = [316170]
Ncells = 8*32
window_start = 148
window_stop = 158
window_size = 8

#Loop through each run
for run_num in runs:
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):
            #Method 1
            data = np.load("calib_waveforms/calib_ped_sub_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            run = data['run']
            asic = data['asic']
            channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            del data

            #Correct samples with pedestal subtraction in mV.
            pedestal = np.percentile(samples[:,:128],50.,axis=1)
            pedestal = pedestal.reshape(len(pedestal),1)
            corr_samples = samples-pedestal

            #initialize a few variables and temp arrays
            Nevents = len(event)         #num of events in samples
            charge = np.zeros(Nevents)
            position = np.zeros(Nevents,dtype=int)
            amplitude = np.zeros(Nevents)
            width = np.zeros(Nevents)

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

                if amplitude[i]>=10.:
                    test_samp = np.zeros((254,3))
                    test_samp[:,0] = corr_samples[i,1:255]
                    test_samp[:,1] = corr_samples[i,0:254]
                    test_samp[:,2] = corr_samples[i,2:256]

                    # Compute PCA
                    pca = PCA(n_components=3)
                    H = pca.fit_transform(test_samp)  # Reconstruct signals based on orthogonal components
                    new_samples = H[:,0]
                    new_samples = np.transpose(new_samples)

                    #determine pulse position
                    position[i] = window_start+np.argmax(new_samples[window_start:window_stop])
                    amplitude[i] = new_samples[position[i]]
                    start_int = int(position[i]-3.)
                    stop_int = int(position[i]+3.)
                    charge[i] = np.sum(new_samples[start_int:stop_int])
                    position[i] = position[i]+1
                    '''
		    plt.figure()
		    models = [test_samp, H]
		    names = ['Observations (mixed signal)',
		   	     'PCA recovered signals']
		    colors = ['red', 'steelblue', 'orange']
		    for ii, (model, name) in enumerate(zip(models, names), 1):
		        plt.subplot(2, 1, ii)
		        plt.title(name)
		        for sig, color in zip(model.T, colors):
			    plt.plot(sig, color=color)
		    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
		    plt.show()
                    '''
                   
            sys.stdout.write('\n')
            print 'Run: ',run_num, ' job time: ', time.time() - start_time

            #Store results in a text file
            print "Saving charge spectrum for run ",run_num," to external file"
            outfile = "charge_spectrum_files/simple_ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            np.savez_compressed(outfile, run=run, asic=asic, channel=channel, event=event, block=block, phase=phase, charge=charge, position=position, amplitude=amplitude, width=width)
            print 'Analysis complete: job time ', time.time() - start_time
