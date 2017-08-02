"""
Miles Winter
Run raw waveforms through transfer function then write out to text file
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

#Run information
#runs = [313348,313349,313350,313351,313352,313353,313369]
runs = np.arange(314391,314416,1)
Ncells = 128

#Loop through each asic
for asic in range(1):
    #Loop through each channel
    for channel in range(14,15):
        #create output file for calibrated samples
        outFile = "calib_waveforms/calib_samples"+str(runs[0])+"_"+str(runs[-1])+"ASIC"+str(asic)+"CH"+str(channel)+".txt"
        f = open(outFile, 'w')
        f.close()
        #Loop through each run
        for run_num in runs:
            #create output file for calibratd samples
            outFile = "calib_waveforms/calib_samples"+str(runs[0])+"_"+str(runs[-1])+"ASIC"+str(asic)+"CH"+str(channel)+".txt"
            f = open(outFile, 'w')
            f.close()

            #Load samples
            data = np.loadtxt("/Users/tmeures/test_suite/runFiles/sampleFileLarge_run"+str(run_num)+"ASIC"+str(asic)+"CH"+str(channel)+".txt")
            Nasic = np.array(data[:,0])
            Nchannel = np.array(data[:,1])
            ievt = np.array(data[:,2])
            blockNumber = np.array(data[:,3])
            blockPhase = np.array(data[:,4])
            samples = np.array(data[:,5:],dtype=int)

            #Load transfer function
            data = np.loadtxt("TF_Tables/TF_table_runs_314392_314413.txt")
            ADC_counts = np.array(data[:,0],dtype=int)
            ADC_to_mV = np.array(data[:,1:])

            #open output file            
            f = open(outFile, 'a')

            #Convert samples from ADC counts to mV w/ transfer function
            samp_size = samples.shape
            mV_samples = np.zeros(samp_size)
            for i in range(samp_size[0]):
                for j in range(samp_size[1]):
                    ADC_index = np.where(ADC_counts==samples[i,j])
                    mV_samples[i,j] = ADC_to_mV[ADC_index[0],j]

                #write to output file
                f.write("%d   " % Nasic[i])
                f.write("%d   " % Nchannel[i])
	        f.write("%d   " % ievt[i])
	        f.write("%d   " % blockNumber[i])
	        f.write("%d   " % blockPhase[i])
                f.write("%.2f  "% mV_samples[i,:])
                f.write("\n")
            f.close()

           
