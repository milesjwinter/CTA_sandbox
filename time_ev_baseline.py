import numpy as np
import matplotlib.pyplot as plt

#Run information
#runs = np.arange(314512,314533,1)
runs = np.arange(314533,314549,1)


mean_value = np.zeros(len(runs))
mean_time = np.zeros(len(runs))
plt.figure(1,figsize=(14,7))
#Loop through each run
for i, run_num in enumerate(runs):
    print "Loading calibrated waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load samples
            data = np.load("run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
            #data = np.load("base_cal_runs/base_cal_samples_Run"+str(run_num)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
            run = data['run']
            asic = data['asic']
            channel = data['channel']
            event = data['event']
            block = data['block']
            phase = data['phase']
            samples = data['samples']
            del data

            times = np.linspace(i*20,(i+1)*20,len(event))
            avg_samples = np.percentile(samples[:,:64],50.,axis=1)
            mean_value[i] = np.mean(avg_samples)
            mean_time[i] = np.mean(times)
            plt.plot(times,avg_samples)


plt.plot(mean_time,mean_value,'k*')
plt.xlabel('Time (minutes)',fontsize=20)
plt.ylabel('Average Measured ADC Counts',fontsize=20)
#plt.xlim([0,Ncells])
#plt.ylim([-10,15])
plt.title('Average Pedestal Subtracted Waveforms',fontsize=24)
#plt.title('Average DC Calibrated Waveforms',fontsize=24)
#plt.xticks(np.arange(0,Ncells+2, 32.0))
plt.show()

