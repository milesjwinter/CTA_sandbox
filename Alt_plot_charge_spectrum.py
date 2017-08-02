import numpy as np
import matplotlib.pyplot as plt
import time
import sys

start_time = time.time()

#run parameters and temp arrays
runs = np.arange(316170,316180,1)
#runs = [316172,316173,316174]
total_charge = np.array([])
total_charge1 = np.array([])
total_charge2 = np.array([])
total_charge3 = np.array([])
total_charge4 = np.array([])
position = np.array([])
position1 = np.array([])
position2 = np.array([])
position3 = np.array([])
position4 = np.array([])
value = np.array([])
value1 = np.array([])
value2 = np.array([])
value3 = np.array([])
value4 = np.array([])

#Loop through each run
for run_num in runs:
    print "Loading charge spectrum: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load charge spectrum
            data = np.load("charge_spectrum_files/charge_spectrum_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            #run = data['run']
            #asic = data['asic']
            #channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            charge = data['charge']
            max_pos = data['max_pos']
            max_val = data['max_val']
            del data

            Nevents = len(event)
	    low_pos, mean_pos, up_pos = np.percentile(max_pos[max_pos>0], [0.,50.,100.])
	    low_val, up_val = np.percentile(max_val[max_val>0],[60.,100.])
            print "Applying data cuts: Asic ",Nasic," Channel ", Nchannel
	    for i in range(Nevents):
                if(i%100==0):
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-100s] %.1f%%" % ('='*int((i)*100./(Nevents)) , (i)*100./(Nevents)))
                    sys.stdout.flush()
                        #if np.count_nonzero(event==event[i])==1:

                if max_val[i]>35:
		#if max_val[i]>=low_val and max_val[i]<=up_val:
		    #if max_pos[i]>=low_pos and max_pos[i]<=up_pos:
                    #if max_pos[i]>=151 and max_pos[i]<=153:
		    if max_pos[i]==153:
			total_charge = np.append(total_charge,charge[i])
			position = np.append(position,max_pos[i])
			value = np.append(value,max_val[i])
                        '''
			if phase[i]==4:
			    total_charge1 = np.append(total_charge1,charge[i])
			    position1 = np.append(position1,max_pos[i])
			    value1 = np.append(value1,max_val[i])
			if phase[i]==12:
			    total_charge2 = np.append(total_charge2,charge[i])
			    position2 = np.append(position2,max_pos[i])
			    value2 = np.append(value2,max_val[i])
			if phase[i]==20:
			    total_charge3 = np.append(total_charge3,charge[i])
			    position3 = np.append(position3,max_pos[i])
			    value3 = np.append(value3,max_val[i])
			if phase[i]==28:
			    total_charge4 = np.append(total_charge4,charge[i])
			    position4 = np.append(position4,max_pos[i])
			    value4 = np.append(value4,max_val[i])
                        '''
	#else:         
	#    print 'more than one event'    
    sys.stdout.write('\n')

print 'Analysis complete: job time ', time.time() - start_time
#print 'Generating histograms:  Number of events ',len(total_charge)

#Plot histogram
plt.figure(1,figsize=(16,8))
plt.hist(total_charge,bins=np.arange(0,1200,4))
plt.title('Charge Spectrum',fontsize=24)
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()

'''
plt.figure(2)
plt.subplot(221)
plt.title('Charge Spectrum',fontsize=24)
plt.hist(total_charge1,bins=np.arange(100,500,2), color='r', alpha=0.3 ,label='phase 1')
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(222)
plt.hist(total_charge2,bins=np.arange(100,500,2), color='b', alpha=0.3, label='phase 2')
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(223)
plt.hist(total_charge3,bins=np.arange(100,500,2), color='g', alpha=0.3, label='phase 3')
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(224)
plt.hist(total_charge4,bins=np.arange(100,500,2), color='y', alpha=0.3, label='phase 4')
plt.xlabel('Charge (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.show()
'''
'''
plt.figure(3)
plt.subplot(221)
plt.title('Position of Max Pulse Height',fontsize=24)
plt.hist(position1,bins=np.arange(140,170,1), color='r', alpha=0.3 ,label='phase 1')
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(222)
plt.hist(position2,bins=np.arange(140,170,1), color='b', alpha=0.3 ,label='phase 2')
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(223)
plt.hist(position3,bins=np.arange(140,170,1), color='g', alpha=0.3 ,label='phase 3')
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(224)
plt.hist(position4,bins=np.arange(140,170,1), color='y', alpha=0.3 ,label='phase 4')
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.show()

plt.figure(4,figsize=(16,8))
plt.hist(position,bins=np.arange(145,170,1))
plt.title('Position of Max Pulse Height',fontsize=24)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()

plt.figure(5)
plt.subplot(221)
plt.title('Maximum Pulse Height',fontsize=24)
plt.hist(value1,bins=np.arange(20,101,1), color='r', alpha=0.3 ,label='phase 1')
plt.xlabel('ADC Counts (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(222)
plt.hist(value2,bins=np.arange(20,101,1), color='b', alpha=0.3 ,label='phase 2')
plt.xlabel('ADC Counts (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(223)
plt.hist(value3,bins=np.arange(20,101,1), color='g', alpha=0.3 ,label='phase 3')
plt.xlabel('ADC Counts (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.subplot(224)
plt.hist(value4,bins=np.arange(20,101,1), color='y', alpha=0.3 ,label='phase 4')
plt.xlabel('ADC Counts (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.grid('on')
plt.legend()
plt.show()
'''
plt.figure(6,figsize=(16,8))
plt.hist(value,bins=np.arange(20,101,1))
plt.title('Maximum Pulse Height',fontsize=24)
plt.xlabel('ADC Counts (mV)',fontsize=20)
plt.ylabel('Counts',fontsize=20)
#plt.yscale('log')
plt.grid('on')
plt.show()
