import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = np.loadtxt('temperature_log.txt')
ind = 793 #last index where temperature sensor was working
run_time = data[:ind,0]
time = data[:ind,1]-data[0,1]
asic0 = data[:ind,2]
asic2 = data[:ind,3]

runs = np.arange(321093,321148,1)
avg_time = np.zeros(len(runs))
avg_ped = np.zeros(len(runs))
avg_0 = np.zeros(len(runs))
avg_2 = np.zeros(len(runs))
#Loop through each run
for i, run_num in enumerate(runs):
    print "Loading waveforms: Run ",run_num
    #Loop through each asic
    for Nasic in range(1):
        #Loop through each channel
        for Nchannel in range(14,15):

            #Load samples
            infile = "run_files/sampleFileAllBlocks_run"+str(run_num)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz"
            data = np.load(infile)
            run = data['run']
            #asic = data['asic']
            #channel = data['channel']
            #event = data['event']
            #block = data['block']
            #phase = data['phase']
            samples = data['samples']
            avg_ped[i] = np.mean(samples[:,:128])
            del data
            
            temp_time = np.array([])
            temp_0 = np.array([])
            temp_2 = np.array([])
            for q in range(len(run_time)):
                if run_time[q]==run_num:
                    temp_time = np.append(temp_time, time[q])
                    temp_0 = np.append(temp_0, asic0[q])
                    temp_2 = np.append(temp_2, asic2[q])

            avg_time[i] = np.mean(temp_time[:8])
            avg_0[i] = np.mean(temp_0[:8])
            avg_2[i] = np.mean(temp_2[:8])

plt.figure()
plt.plot(time/3600.,asic0,label='ASIC0')
plt.plot(time/3600.,asic2,label='ASIC2')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (C$^\cdot$)')
plt.minorticks_on()
#plt.xlim([4,6])
plt.title('Asic Temperature vs Time')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/mileswinter/Documents/UW/Research/CTA/Presentations/2017_07_20/time_vs_temp.png')
plt.show()

plt.figure()
plt.plot(avg_time/3600.,avg_ped)
plt.xlabel('Time (hours)')
plt.ylabel('Avg. Pedestal (ADC Counts)')
plt.minorticks_on()
plt.title('Average Pedestal vs Time')
plt.tight_layout()
plt.savefig('/Users/mileswinter/Documents/UW/Research/CTA/Presentations/2017_07_20/ped_vs_time.png')
plt.show()

plt.figure()
plt.plot(avg_ped,avg_0,label='ASIC0')
plt.plot(avg_ped,avg_2,label='ASIC2')
plt.xlabel('Avg. Pedestal (ADC Counts)')
plt.ylabel('Temperature (C$^\cdot$)')
plt.minorticks_on()
#plt.xlim([4,6])
plt.title('Avg. Asic Temperature vs Avg. Pedestal')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/mileswinter/Documents/UW/Research/CTA/Presentations/2017_07_20/temp_vs_ped.png')
plt.show()

plt.figure()
plt.plot(avg_0,avg_ped,label='ASIC0')
plt.plot(avg_2,avg_ped,label='ASIC2')
plt.xlabel('Avg. Pedestal (ADC Counts)')
plt.ylabel('Temperature (C$^\cdot$)')
plt.minorticks_on()
#plt.xlim([4,6])
plt.title('Avg. Pedestal vs Avg. Asic Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/mileswinter/Documents/UW/Research/CTA/Presentations/2017_07_20/ped_vs_temp.png')
plt.show()


