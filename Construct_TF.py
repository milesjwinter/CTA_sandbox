#construct Tranfer Function
#Run numbers 313344-313369

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

start_time = time.time()

#Run information
runs = np.arange(313344,313369,1)
vpeds = np.arange(125,4062,164)
trans_func = np.zeros((len(runs),64))

#Loop through each run
for run_num in runs:
    #Loop through each asic
    for asic in range(1):
        #Loop through each channel
        for channel in range(1):
            #Load baseline
            data = np.loadtxt("run_files/flasher_av_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            baseline = 0.
            for i in range(1,len(data[:,0])):
                if data[i,4]>0.0:
                    baseline = np.array(data[i,4:], dtype=int)
            trans_func[run_num-runs[0],:] = baseline

#perform linear interpolation and construct lookup table
trans_func_table = np.zeros((3750-290,65))
points = np.arange(290,3750,1,dtype=int)
trans_func_table[:,0] = points
mV_vpeds = 0.609*vpeds+26.25
for i in range(64):
    ADC_to_mV = interp1d(trans_func[:,i],mV_vpeds,fill_value='extrapolate')
    trans_func_table[:,i+1] = ADC_to_mV(points)


np.savetxt("transfer_function_table.txt", trans_func_table,fmt='%.2f' )

'''
mV_vpeds = 0.609*vpeds+26.25
plt.figure(1)
for i in range(64):
    plt.plot(mV_vpeds,trans_func[1+i,:])
plt.title('DC Transfer Function')
plt.xlabel('Input DC Voltage (mV)')
plt.ylabel('Measured ADC Counts')
plt.xlim([100,2500])
plt.show()
'''
