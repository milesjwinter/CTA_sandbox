import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import time
import sys
from scipy.stats import gaussian_kde

start_time = time.time()
#Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2.*sigma**2))

#run parameters and temp arrays
runs = np.arange(320247,320251,1)
#runs = np.arange(320235,320244,1)
#runs = np.append(runs,np.arange(320245,320247,1))
#runs = np.arange(319428,319435,1)
#runs = np.arange(319518,319558,1)
#runs = np.arange(319520,319530,1)
#runs = np.arange(319408,319418,1)
#runs = np.append(runs,np.arange(319436,319446,1))
#runs = np.arange(318745,318765,1)
#runs = np.arange(319149,319162,1)
#runs = [319160,319161]
#runs = np.arange(318775,318782,1)
#runs = np.arange(316170,316180,1)
#runs = np.arange(319252,319272,1)
#runs = [319270]
#runs = np.arange(319210,319214,1)
#runs = np.arange(317520,317536,1)
#runs = [316170,316171,316172,316173,316174]
#runs = np.arange(314490,314500,1)
#runs = np.append(runs,np.arange(314513,314533,1))
#runs = np.arange(314515,314533,1)
#runs = [314530,314531,314532]
total_charge = np.array([])
total_position = np.array([])
total_amplitude = np.array([])
total_fit_charge = np.array([])
total_fit_position = np.array([])
total_fit_amplitude = np.array([])
total_fit_width = np.array([])
total_ped_max = np.array([])
select_bin = 194
select_phase = 28
Ncells = 256
data = np.loadtxt('par_list_newest_mv.txt')
const = data[:,0]
mean = data[:,1]
sigma = data[:,2]
 
#Loop through each run
for run_num in runs:
    print "Loading charge spectrum: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load charge spectrum
            #infile = "charge_spectrum_files/simple_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile = "charge_spectrum_files/simple_base_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz" 
            #infile = "charge_spectrum_files/simple_ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile = "charge_spectrum_files/simple_alt_fit_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile = "charge_spectrum_files/dc_cal_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile = "charge_spectrum_files/alt_ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            infile = "charge_spectrum_files/ped_sub_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            #infile = "charge_spectrum_files/ped_sub_dark_charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz"
            data = np.load(infile)
            #run = data['run']
            #asic = data['asic']
            #channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            charge = data['charge']
            position = data['position']
            amplitude = data['amplitude']
            ped_max = data['ped_max']
            ped_min = data['ped_min']
            
            #fit_charge = data['fit_charge']
            #fit_position = data['fit_position']
            #fit_amplitude = data['fit_amplitude']
            #fit_width = data['fit_width']
            del data

            print "Number of Events: ",len(event)

	    temp_charge = np.array([])
	    temp_position = np.array([])
	    temp_amplitude = np.array([])
            temp_fit_charge = np.array([])
            temp_fit_position = np.array([])
            temp_fit_amplitude = np.array([])
	    temp_fit_width = np.array([])
            temp_ped_max = np.array([])

            Nevents = len(event)
	    #low_pos, up_pos = np.percentile(position, [0.,50.])
	    #low_amp, up_amp = np.percentile(amplitude,[0.,100.])
            print "Applying data cuts: Asic ",Nasic," Channel ", Nchannel
	    for i in range(Nevents):
                if(i%100==0):
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-100s] %.1f%%" % ('='*int((i)*100./(Nevents)) , (i)*100./(Nevents)))
                    sys.stdout.flush()
                if charge[i] > -100.:
                #if charge[i] == 0.:
	            #if amplitude[i]>0:
                    #if width[i]>1. and width[i]<6.:
                    if ped_max[i]<=10.:# and ped_min[i]>=-10.:
		        #if position[i]>=132 and position[i]<=134:
                        #if 192 <= position[i] <= 194:
                        #if position[i]>=152 and position[i]<=153:
                        #if position[i]>=38 and position[i]<=46:
		        #if position[i]==select_bin: 
                        #if position[i] in [192,194]:
                            #if phase[i]==select_phase:
                            

			    temp_charge = np.append(temp_charge,charge[i])
			    #total_charge = np.append(total_charge,np.sqrt(2.*np.pi)*amplitude[i]*np.random.normal(5, .9))
			    temp_position = np.append(temp_position,position[i])
			    temp_amplitude = np.append(temp_amplitude,amplitude[i])
			    temp_ped_max = np.append(temp_ped_max,ped_max[i])
			    #temp_fit_charge = np.append(temp_fit_charge,fit_charge[i])
			    #temp_fit_position = np.append(temp_fit_position,fit_position[i])
			    #temp_fit_amplitude = np.append(temp_fit_amplitude,fit_amplitude[i])
			    #temp_fit_width = np.append(temp_fit_width,fit_width[i])

            sys.stdout.write('\n')
            total_charge = np.append(total_charge,temp_charge)
            total_position = np.append(total_position,temp_position)
            total_amplitude = np.append(total_amplitude,temp_amplitude)
            total_ped_max = np.append(total_ped_max,temp_ped_max)
            #total_fit_charge = np.append(total_fit_charge,temp_fit_charge)
            #total_fit_position = np.append(total_fit_position,temp_fit_position)
            #total_fit_amplitude = np.append(total_fit_amplitude,temp_fit_amplitude)
            #total_fit_width = np.append(total_fit_width,temp_fit_width)

print 'Analysis complete: job time ', time.time() - start_time

minorLocator = AutoMinorLocator()
points = np.arange(-50,600,1)

#Plot histogram
np.savetxt('total_charge_newest_mv.txt',total_charge*0.609,fmt='%f')

fig, ax = plt.subplots(figsize=(14,7))
#fig, ax = plt.figure(1,figsize=(16,8))
for i in range(len(const)):
    plt.plot(points,gauss_function(points,const[i],mean[i],sigma[i]),'r',lw=2,linestyle='dashed')
plt.hist(total_charge*0.609,bins=np.arange(-60,800,4*0.609),edgecolor='k')
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV$\cdot$ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()


fig, ax = plt.subplots(figsize=(14,7))
plt.hist(total_charge*0.609,bins=np.arange(-60,800,4*0.609),edgecolor='k')
#plt.hist(total_charge,bins=np.arange(-20,400,2),edgecolor='k')
#plt.hist(total_charge,bins=np.arange(-200,200,2),edgecolor='k')
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV$\cdot$ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()


'''
fig, ax = plt.subplots(figsize=(16,8))
#fig, ax = plt.figure(1,figsize=(16,8))
plt.subplot(211)
plt.hist(total_charge,bins=np.arange(0,1000,4),color='b',label='No Fit')
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.legend()
plt.subplot(212)
plt.hist(total_fit_charge,bins=np.arange(0,1000,4),color='g',label='Gaussian Fit')
#plt.title('Charge Spectrum w/ Gaussian Fit',fontsize=24)
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.legend()
plt.show()
'''

fig, ax = plt.subplots(figsize=(14,7))
#plt.figure(2,figsize=(16,8))
plt.hist(total_position,bins=np.arange(0,Ncells,1),edgecolor='k')
plt.title('Position of Max Pulse Height',fontsize=24)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
plt.xlim([0,Ncells])
plt.xticks(np.arange(0,Ncells+2, 32.0))
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()


'''
fig, ax = plt.subplots(figsize=(16,8))
#plt.figure(2,figsize=(16,8))
plt.subplot(211)
plt.hist(total_position,bins=np.arange(145,160,1),color='b',label='Pos. of max value')
plt.title('Position of Max Pulse Height',fontsize=24)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.legend()
plt.subplot(212)
plt.hist(total_fit_position,bins=np.arange(145,160,1), color='g',label='Fit mean value')
#plt.title('Position of Max Pulse Height',fontsize=24)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.legend()
plt.show()
'''

fig, ax = plt.subplots(figsize=(14,7))
#plt.figure(3,figsize=(16,8))
plt.hist(total_amplitude*0.609,bins=np.arange(-10,100,2*0.609),edgecolor='k')
plt.title('Maximum Pulse Amplitude',fontsize=24)
plt.xlabel('Amplitude (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()
'''
print np.amax(total_ped_max)
fig, ax = plt.subplots(figsize=(14,7))
#plt.figure(3,figsize=(16,8))
plt.hist(total_ped_max,bins=np.arange(0,100,1),edgecolor='k')
plt.title('Maximum Baseline Amplitude',fontsize=24)
plt.xlabel('Amplitude (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.show()
'''
'''
fig, ax = plt.subplots(figsize=(16,8))
#plt.figure(3,figsize=(16,8))
plt.subplot(211)
plt.hist(total_amplitude,bins=np.arange(0,200,2),color='b',label='Max value')
plt.title('Maximum Pulse Amplitude',fontsize=24)
plt.xlabel('Amplitude (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.legend()
plt.subplot(212)
plt.hist(total_fit_amplitude,bins=np.arange(0,200,2),color='g',label='Fit normalization')
#plt.title('Maximum Pulse Amplitude',fontsize=24)
plt.xlabel('Amplitude (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
#Format tick marks
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
plt.tick_params(which='major', length=7, width=2)
plt.tick_params(which='minor', length=4)
plt.minorticks_on()
plt.grid('off')
plt.legend()
plt.show()
'''

'''
plt.figure(4,figsize=(16,8))

# Calculate the point density
y = total_amplitude
x = total_width
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
'''
'''
plt.figure(4,figsize=(16,8))
plt.hist2d(total_fit_width,total_fit_amplitude,bins=(np.arange(0,8.2,0.2),np.arange(0,150,1)),cmap=plt.cm.jet)
#plt.scatter(x, y, c=z, s=50, edgecolor='')
plt.title('Pulse Amplitude vs Width',fontsize=24)
plt.ylabel('Amplitude (mV)',fontsize=20)
plt.xlabel('Width (ns)',fontsize=20)
#plt.yscale('log')
plt.colorbar()
plt.grid('on')
plt.show()
'''
'''
plt.figure(4,figsize=(16,8))
plt.hist2d(total_position,total_amplitude,bins=(np.arange(145,161,1),np.arange(0,151,10)),cmap=plt.cm.jet)
#plt.scatter(x, y, c=z, s=50, edgecolor='')
plt.title('Pulse Amplitude vs Position',fontsize=24)
plt.ylabel('Amplitude (mV)',fontsize=20)
plt.xlabel('Time (ns)',fontsize=20)
#plt.yscale('log')
plt.colorbar()
#plt.grid('on')
plt.show()
'''
