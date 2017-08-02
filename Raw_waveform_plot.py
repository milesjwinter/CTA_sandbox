import numpy as np
import os, sys, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#plt.ion()

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

#Run information
#runs = [318784]
#runs = [314512]
#runs = np.arange(317433,317436,1)
#runs = np.arange(319428,319435,1)
#runs = np.arange(316170,316180,1)
#runs = [314547,314548]
runs = [320246,320254]
#runs = [314494]
#runs = [317052]
#runs = np.arange(317052,317055,1)
#runs = [316151]
#runs = [316173]
#runs = [314499]
#runs = [314514]
#runs = [314518]
#runs = [314540]
#runs = [319165]
#runs = np.arange(319162,319172,1)
#runs = [317109]
#runs = np.arange(316170,316180,1)
#runs = [317439,317440,317443,317445,317446,317448,317450,317452,317454,317455,317456,317457,317458,317459]
#runs = [317467,317468,317469,317470,317471,317472,317473,317474]
#runs = [317115,317118,317123,317127,317129,317130,317133,317138,317141,317144,317148,317152,317153,317154,317155]
#runs = np.arange(317516,317536,1)
#runs = np.arange(317520,317536,1)
#runs = np.array([317433,317434,317435,317437])
#runs = np.arange(317505,317515,1)
#runs = np.arange(314490,314500,1)
#runs = np.arange(314513,314533,1)
Nblocks = 8
Ncells = Nblocks*32

#load and calculate baseline waveform
#data = np.load("run_files/sampleFileAllBlocks_run316175ASIC0CH14.npz")
#data = np.load("run_files/sampleFileAllBlocks_run317052ASIC0CH14.npz")
data = np.load("run_files/sampleFileAllBlocks_run316151ASIC0CH14.npz")
raw_samples = data['samples']
print raw_samples.shape
adj_base = raw_samples-np.mean(raw_samples,axis=1).reshape(len(raw_samples),1)
#baseline = np.mean(adj_base,axis=0)
baseline = np.mean(raw_samples,axis=0)
cal_samples = raw_samples-baseline
del data

all_std1 = np.array([])
all_std2 = np.array([])
all_std3 = np.array([])
all_std4 = np.array([])
all_std5 = np.array([])
all_std6 = np.array([])
all_std7 = np.array([])
all_std8 = np.array([])
select_samples = np.array([])
all_amp_vals = np.array([])
#Loop through each run
for i, run_num in enumerate(runs):
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load samples
            infile = "run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            #infile = "base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile="calib_waveforms/calib_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            data = np.load(infile)
            #run = data['run']
            #asic = data['asic']
            #channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            print samples.shape
            del data
            print np.unique(phase)
            print np.unique(block)
            #initialize a few variables and temp arrays
            Nevents = len(event)         #num of events in samples

            #Correct samples with pedestal subtraction in mV.
            #adj_samples = samples-np.mean(samples[:,:64],axis=1).reshape(len(samples),1)
            #corr_samples = adj_samples-np.mean(adj_samples,axis=0)
            #adj_samples = samples-baseline
            #corr_samples = adj_samples-np.mean(adj_samples[:,:64],axis=1).reshape(len(adj_samples),1)
            #corr_samples = samples-baseline
            #corr_samples = samples-np.mean(samples,axis=0)
            corr_samples = samples
            #select_samples = np.array([])
            #corr_samples = samples-np.mean(samples[:,:64],axis=1).reshape(len(samples),1)
            #charge = np.sum(corr_samples[:,128:137],axis=1)
            print Nevents
            #index = np.random.permutation(Nevents).astype(int)
            
            plt.figure(figsize=(12,5))
            count = 0
            #for j in range(Nevents):
            for j in range(0,350):
                #if block[j]==0 and phase[j]==4:
                #if np.absolute(charge[j])<50:
                if np.amin(samples[j])>100.:
                    count += 1.
                    plt.plot(np.arange(0,Ncells,1),corr_samples[j,:Ncells],'b',lw=0.5,alpha=0.6)
                    #plt.plot(np.arange(0,Ncells,1),samples[j,:Ncells],'b',lw=0.5,alpha=0.6)
                    if count==50.:
                        #plt.plot(np.arange(0,Ncells,1),baseline[:Ncells]-baseline[:Ncells],'r',lw=1.5,alpha=1)
                        #plt.plot(np.arange(0,Ncells,1),baseline[:Ncells],'r',lw=1.5,alpha=1)
                        plt.xlabel('Time (ns)',fontsize=20)
                        #plt.ylabel('Measured ADC Counts',fontsize=20)
                        plt.ylabel('Amplitude (ADC Counts)',fontsize=20)
                        plt.xlim([0,Ncells])
                        #plt.ylim([-20,50])
                        plt.xticks(np.arange(0,Ncells+2, 32.0))
                        plt.show()
                        break
                      
            '''
            plt.figure()
            plt.hist(charge)
            plt.show()

            plt.figure()
            plt.hist(np.argmax(corr_samples,axis=1),bins=np.arange(0,256,1))
            plt.show()
            '''
            
            all_amp_vals = np.append(all_amp_vals, np.amax(corr_samples[:,:Ncells],axis=1)*0.609)
            
            for j in range(Nevents):
                if block[j]==0 and phase[j]==12:
            
		    #all_std1 = np.append(all_std1,corr_samples[j,:32].flatten())
		    #all_std2 = np.append(all_std2,corr_samples[j,32:64].flatten())
		    #all_std3 = np.append(all_std3,corr_samples[j,64:96].flatten())
		    #all_std4 = np.append(all_std4,corr_samples[j,96:128].flatten())
		    #all_std5 = np.append(all_std5,corr_samples[j,128:160].flatten())
		    #all_std6 = np.append(all_std6,corr_samples[j,160:192].flatten())
		    #all_std7 = np.append(all_std7,corr_samples[j,192:224].flatten())
		    #all_std8 = np.append(all_std8,corr_samples[j,224:256].flatten())
                    select_samples = np.append(select_samples,corr_samples[j,:])
            
            '''
            #yhist, xhist = np.histogram(corr_samples.flatten(), bins = np.arange(-15,15,1))
            yhist, xhist = np.histogram(select_samples, bins = np.arange(-15,15,1))
            popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
            plt.figure()
            #plt.hist(corr_samples.flatten(),bins=np.arange(-15,15,1))
            plt.hist(select_samples,bins=np.arange(-15,15,1))
            i = np.arange(-15, 15, 0.1)
            plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
            plt.xlabel('ADC Channel',fontsize=14)
            plt.ylabel('Counts',fontsize=14)
            #plt.ylim([0,250000])
            plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
            plt.text(-14,1.4e3,'All Blocks',fontsize=12)
            plt.text(-14, 1.3e3, '$\sigma$=%.2f'%np.std(corr_samples.flatten()), fontsize=12)
            plt.show()
            '''
	    #print count    


	    '''
	    label = 'plots/New_sync_waveforms/raw_waveforms/HV_on_LED_On_old/raw_waveforms_run_%s.png' % run_num
	    #plt.figure(figsize=(12,5))
	    fig, ax = plt.subplots(figsize=(16,8))
	    plt.subplot(211)
	    for i in range(50):
		plt.plot(np.arange(0,Ncells,1),samples[index[i],:])
	    plt.xlabel('Time (ns)',fontsize=20)
	    plt.ylabel('Measured ADC Counts',fontsize=20)
	    plt.xlim([0,Ncells])
	    #plt.ylim([-10,15])
	    plt.title('Raw Waveforms: Run %s' % run_num,fontsize=24)
	    plt.xticks(np.arange(0,Ncells+2, 32.0))
	    plt.minorticks_on()
	    plt.subplot(212)
	    for i in range(50):
		plt.plot(np.arange(0,Ncells,1),corr_samples[index[i],:])
	    plt.xlabel('Time (ns)',fontsize=20)
	    plt.ylabel('Measured ADC Counts',fontsize=20)
	    plt.xlim([0,Ncells])
	    #plt.ylim([-10,15])
	    plt.title('Pedestal Subtracted Waveforms: Run %s' % run_num,fontsize=24)
	    plt.xticks(np.arange(0,Ncells+2, 32.0))
	    plt.minorticks_on()
	    plt.tight_layout()
	    plt.savefig(label,dpi=200)
	    plt.show()
	    plt.close()
	    '''
'''
up_lim = 70.
cal_amp_vals = np.amax(cal_samples[:,:Ncells],axis=1)*0.609
yhist, xhist = np.histogram(all_amp_vals, bins = np.arange(0,up_lim,0.5))
ychist, xchist = np.histogram(cal_amp_vals, bins = np.arange(0,up_lim,0.5))
dif_amp_vals = yhist-ychist
popt, pcov = curve_fit(gaussian, np.arange(0.25,up_lim-0.5,.5), yhist, [1000, 5, 1])
plt.figure()
#plt.hist(all_amp_vals,bins=np.arange(0,up_lim,.5))
plt.hist(dif_amp_vals[dif_amp_vals>=0],bins=np.arange(0,up_lim,.5))
#i = np.arange(0, up_lim, 0.1)
#plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.text(10,6000,'$\mu=%.2f$'%popt[1])
plt.text(10,5500,'$\sigma=%.2f$'%popt[2])
plt.xlabel('Max Amplitude (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.show()




#yhist, xhist = np.histogram(corr_samples.flatten(), bins = np.arange(-15,15,1))
yhist, xhist = np.histogram(select_samples, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.figure()
#plt.hist(corr_samples.flatten(),bins=np.arange(-15,15,1))
plt.hist(select_samples,bins=np.arange(-15,15,1))
i = np.arange(-15, 15, 0.1)
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.text(-14,7.2e3,'All Blocks',fontsize=12)
plt.text(-14, 6.8e3, '$\sigma$=%.2f'%np.std(corr_samples.flatten()), fontsize=12)
plt.show()
'''
'''
#yhist, xhist = np.histogram(all_std1, bins = np.arange(-15,15,1))
#popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
#plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
i = np.arange(-15, 15, 0.1)
#plt.figure()
fig, ax = plt.subplots(figsize=(12,6))
plt.subplot(241)
plt.hist(all_std1,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std1, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.text(-14,1.1e3,'Block 1',fontsize=12)
plt.text(-14, 1e3, '$\sigma$=%.2f'%np.std(all_std1), fontsize=12)
plt.subplot(242)
plt.hist(all_std2,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std2, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.text(-14,1.1e3,'Block 2',fontsize=12)
plt.text(-14, 1.e3, '$\sigma$=%.2f'%np.std(all_std2), fontsize=12)
plt.subplot(243)
plt.hist(all_std3,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std3, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.text(-14,1.1e3,'Block 3',fontsize=12)
plt.text(-14, 1e3, '$\sigma$=%.2f'%np.std(all_std3), fontsize=12)
plt.subplot(244)
plt.hist(all_std4,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std4, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.text(-14,1.1e3,'Block 4',fontsize=12)
plt.text(-14, 1e3, '$\sigma$=%.2f'%np.std(all_std4), fontsize=12)
plt.subplot(245)
plt.hist(all_std5,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std5, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.text(-14,1.1e3,'Block 5',fontsize=12)
plt.text(-14, 1e3, '$\sigma$=%.2f'%np.std(all_std5), fontsize=12)
plt.subplot(246)
plt.hist(all_std6,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std6, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.text(-14,1.1e3,'Block 6',fontsize=12)
plt.text(-14, 1e3, '$\sigma$=%.2f'%np.std(all_std6), fontsize=12)
plt.subplot(247)
plt.hist(all_std7,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std7, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.text(-14,1.1e3,'Block 7',fontsize=12)
plt.text(-14, 1e3, '$\sigma$=%.2f'%np.std(all_std7), fontsize=12)
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.subplot(248)
plt.hist(all_std8,bins=np.arange(-15,15,1))
yhist, xhist = np.histogram(all_std8, bins = np.arange(-15,15,1))
popt, pcov = curve_fit(gaussian, np.arange(-14.5,14.5,1), yhist, [100000, 0, 2])
plt.plot(i, gaussian(i, *popt),'r',linestyle='dashed')
plt.xlabel('ADC Channel',fontsize=14)
plt.ylabel('Counts',fontsize=14)
#plt.ylim([0,250000])
plt.ylim([0,1200])
plt.text(-14,1.1e3,'Block 8',fontsize=12)
plt.text(-14, 1.e3, '$\sigma$=%.2f'%np.std(all_std8), fontsize=12)
plt.ticklabel_format(axis='y',style='sci', scilimits=(5,6))
plt.tight_layout()
plt.show()
'''
