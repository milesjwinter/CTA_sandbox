import numpy as np
import matplotlib.pyplot as plt
import time
import sys

start_time = time.time()

argList = sys.argv

if(len(argList)>2):
        run_start = int(argList[1])
        run_stop = int(argList[2])
elif(len(argList)>1):
        run_start = int(argList[1])
        run_stop = run_start
else:
        start = 292836
        stop = 292836
print argList
print run_start, run_stop

#Run information
Nasic = 0
Nchannel = 14
vped = 1106.
Ncells = 256
Nruns = run_stop-run_start+1

#Load samples
print "Loading Waveforms ...."
#data = np.loadtxt("calib_waveforms/calib_samples_"+str(run_start)+"_"+str(run_stop)+"_ASIC"+str(asic)+"_CH"+str(channel)+".txt")
#data = np.load("base_cal_runs/base_cal_samples_Run"+str(run_start)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
#data = np.load("calib_waveforms/calib_samples_Run"+str(run_start)+"_ASIC"+str(Nasic)+"_CH"+str(Nchannel)+".npz")
data = np.load("run_files/sampleFileAllBlocks_run"+str(run_start)+"ASIC"+str(Nasic)+"CH"+str(Nchannel)+".npz")
run = np.array(data['run'])
asic = np.array(data['asic'])
channel = np.array(data['channel'])
event = np.array(data['event']) 
blocknum = np.array(data['block']) 
blockphase = np.array(data['phase'])
samples = np.array(data['samples'])*0.609
times = np.arange(0,Ncells,1)
print "Plotting Waveforms ...."
#Pulse_average = np.mean(samples,axis=0)
plt.figure(1,figsize=(12,5))
for i in range(0,200):#range(len(samples[:,0])):
    corr_samples = samples[i,:]-np.mean(samples[i,:24])
    #corr_samples = samples[i,:]
    #if blockphase[i]==2:
        #if np.amax(corr_samples[:24])<5. and np.amin(corr_samples[:24])>-5.:
        #plt.plot(times,samples[i,:]-np.mean(samples[:,:128]),lw=1.,alpha=0.8)
    #plt.plot(times,corr_samples,'b',lw=0.8,alpha=0.8)
    plt.plot(times,samples[i],'b',lw=0.8,alpha=0.8)
plt.xlabel('Time (ns)',fontsize=20)
plt.ylabel('mV',fontsize=20)
#plt.title('DC Calibrated Waveforms',fontsize=24)
plt.xlim([0,Ncells])
#plt.xlim([128,192])
#plt.xticks(np.arange(0,Ncells+2, 32.0))
#plt.minorticks_on()
#plt.grid('on')
plt.show()

