import numpy as np
import os, sys, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.misc import factorial

def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def negLogLikelihood(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


#Run information
#runs = [316175]
#runs = np.arange(316175,316178,1)
#runs = [314540]
runs = [317050,317052,317053,317054,316175,316176,316177,316178,316179]
#runs = np.arange(317052,317055,1)
#runs = [316151]
#runs = [314498]
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
#runs = np.arange(317050,317055,1)
#runs = np.arange(317484,317491,1)
Nblocks = 4
Ncells = Nblocks*32
start = 0
stop = 128
delta_t = float(stop-start)
'''
#load and calculate baseline waveform
#data = np.load("run_files/sampleFileAllBlocks_run316175ASIC0CH14.npz")
data = np.load("run_files/sampleFileAllBlocks_run317052ASIC0CH14.npz")
#data = np.load("run_files/sampleFileAllBlocks_run316151ASIC0CH14.npz")
raw_samples = data['samples']
adj_base = raw_samples-np.mean(raw_samples,axis=1).reshape(len(raw_samples),1)
#baseline = np.mean(adj_base,axis=0)
baseline = np.mean(raw_samples,axis=0)
cal_samples = raw_samples-baseline
del data
'''
all_amp_vals = np.array([])
all_cal_amp_vals = np.array([])
all_charge_vals = np.array([])
all_cal_charge_vals = np.array([])
all_pos_vals = np.array([])
all_cal_pos_vals = np.array([])
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
            del data
            #initialize a few variables and temp arrays
            Nevents = len(event)         #num of events in samples
            print Nevents

            #Correct samples with pedestal subtraction in mV.
            #corr_samples = samples[:,start:stop]-np.mean(samples[:,start:stop],axis=0)
            corr_samples = samples[:,start:stop]-np.percentile(samples[:,start:stop],50.,axis=0)
            if run_num<317000:
                all_amp_vals = np.append(all_amp_vals, np.amax(corr_samples,axis=1))
                all_pos_vals = np.append(all_pos_vals, np.argmax(corr_samples,axis=1))
                all_charge_vals = np.append(all_charge_vals, np.sum(corr_samples[:,:],axis=1))
            else:
                all_cal_amp_vals = np.append(all_cal_amp_vals, np.amax(corr_samples,axis=1))
                all_cal_pos_vals = np.append(all_cal_pos_vals, np.argmax(corr_samples,axis=1))
                all_cal_charge_vals = np.append(all_cal_charge_vals, np.sum(corr_samples[:,:],axis=1))

# minimize the negative log-Likelihood
data = all_cal_amp_vals
#data = all_amp_vals
cut_off = np.percentile(all_cal_amp_vals,99.87)
total_counts = np.sum(all_amp_vals>=cut_off)
print total_counts
print len(all_amp_vals)
print cut_off
'''
plt.figure()
for i in range(2000):
    if int(all_cal_amp_vals[i])==4:
        plt.plot(all_cal_pos_vals[i],all_cal_amp_vals[i],'*')
plt.show()
'''
'''
result = minimize(negLogLikelihood,  # function to minimize
                  x0=np.ones(1),     # start value
                  args=(data,),      # additional arguments for function
                  method='Powell',   # minimization method, see docs
                  )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
print(result)
# plot poisson-deviation with fitted parameter
x_plot = np.linspace(0, 20, 1000)
plt.figure()
plt.hist(data, bins=np.arange(15) - 0.5, normed=True)
plt.plot(x_plot, poisson(x_plot, result.x), 'r-', lw=2)
plt.show()
'''
up_lim=50.
plt.figure()
# the bins should be of integer width, because poisson is an integer distribution
entries, bin_edges, patches = plt.hist(all_cal_amp_vals, np.arange(0,up_lim,1.0), normed=True,alpha=0.7,label='HV off')
centries, cbin_edges, cpatches = plt.hist(all_amp_vals, np.arange(0,up_lim,1.0), normed=True,alpha=0.7, label='HV on')
# calculate binmiddles
bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

# fit with curve_fit
#parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 
parameters, cov_matrix = curve_fit(gaussian, bin_middles, entries)
# plot poisson-deviation with fitted parameter
x_plot = np.linspace(0, up_lim, 1000)

#plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2)
#plt.plot(x_plot, gaussian(x_plot, *parameters), 'r-', lw=2)
#plt.axvline(np.percentile(all_cal_amp_vals,84.13),color='k',lw=0.8,linestyle='dashed')
plt.axvline(np.percentile(all_cal_amp_vals,97.72),color='k',lw=0.9,linestyle='dotted', label='97.7th Perc.')
plt.axvline(np.percentile(all_cal_amp_vals,99.87),color='k',lw=0.9,linestyle='dashed',label='99.9th Perc.')
#plt.text(12,17500,'$\mu=%.2f$'%parameters[1])
#plt.text(20,0.3,'$3\sigma=%.2f$'%np.percentile(all_cal_amp_vals,99.87))
#plt.text(10,15500,'99.9 perc.=%.2f'%np.percentile(data,99.9))
plt.xlabel('Max Amplitude (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.legend()
plt.show()

cut_off = np.percentile(all_cal_amp_vals,99.87)
cut_off2 = np.percentile(all_cal_amp_vals,97.72)
total_counts = np.sum(all_amp_vals>=cut_off2)
print total_counts
print len(all_amp_vals)
print '3 sigma ',cut_off
print '2 sigma ', np.percentile(all_cal_amp_vals,97.72), ' , ', np.percentile(all_cal_amp_vals,84.13)
print '1 sigma ', np.percentile(all_cal_amp_vals,84.13)
'''
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
'''
print np.sum(all_cal_charge_vals)
print np.sum(all_charge_vals)
print (np.sum(all_charge_vals)-np.sum(all_cal_charge_vals))/(delta_t*1.e-9*47.)
print np.percentile(all_cal_charge_vals,50.),'   ',np.mean(all_cal_charge_vals)
print np.percentile(all_charge_vals,50.),'   ',np.mean(all_charge_vals)

cal_shift = np.percentile(all_cal_charge_vals,50.)
all_shift = np.percentile(all_charge_vals,50.)
'''
plt.figure()
plt.hist(all_cal_charge_vals, np.arange(-600,1000,10.0), normed=True,alpha=0.7,label='HV off')
plt.axvline(np.percentile(all_cal_charge_vals,50.),color='k',lw=0.9,linestyle='dashed',label='Median HV off')
plt.hist(all_charge_vals, np.arange(-600,1000,10.0), normed=True,alpha=0.7,label='HV on')
#plt.xlabel('Charge (mV$\cdot$ns)',fontsize=20)
plt.axvline(np.percentile(all_charge_vals,50.),color='k',lw=0.9,linestyle='dotted',label='Median HV on')
plt.xlabel('Charge (ADC$\cdot$ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.legend()
plt.show()
'''
print 'max ', np.amax(all_charge_vals/(47.))
print 'min ',np.amin(all_charge_vals/(47.))
mean_cal_charge = np.mean(all_cal_charge_vals/(47.))
mean_charge = np.mean(all_charge_vals/(47.))
dark_rate  = (mean_charge-mean_cal_charge)/(128.e-9)
print (mean_charge-mean_cal_charge)*47.
print 'dark charge: ',dark_rate

plt.figure()
centries, cbin_edges, cpatches = plt.hist(all_cal_charge_vals/(47.),np.arange(-35,85,1.0), normed=True,cumulative=False,alpha=0.7,label='HV off')
#plt.axvline(np.percentile(all_cal_charge_vals,50.),color='k',lw=0.9,linestyle='dashed',label='Median HV off')
entries, bin_edges, patches = plt.hist(all_charge_vals/(47.), np.arange(-35,85,1.0), normed=True,cumulative=False,alpha=0.7,label='HV on')
#bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
#plt.step(bin_middles,entries-centries,where='mid',color='k')
#plt.xlabel('Charge (mV$\cdot$ns)',fontsize=20)
plt.axvline(mean_cal_charge,color='k',lw=0.9,linestyle='dashed',label='Mean HV off')
plt.axvline(mean_charge,color='k',lw=0.9,linestyle='dotted',label='Mean HV on')
plt.xlim([-20,30])
plt.xlabel('PE',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.legend()
plt.show()
