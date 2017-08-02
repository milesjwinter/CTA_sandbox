import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

data = np.loadtxt('temperature_logs/temp_voltage_log.txt')
time = data[:,1]-data[0,1]
voltage = data[:,2]
asic0 = data[:,3]
asic2 = data[:,4]

'''
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
#plt.savefig('/Users/mileswinter/Documents/UW/Research/CTA/Presentations/2017_07_20/time_vs_temp.png')
plt.show()
'''

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(2, 2)
ax0 = plt.subplot(gs[0])
ax0.plot(time/3600.,voltage*1000.)
ax0.set_xlabel('Time (hours)')
ax0.set_ylabel('Voltage (mV)')
ax0.minorticks_on()
ax2 = plt.subplot(gs[2])
ax2.plot(time/3600.,asic0,label='ASIC0')
ax2.plot(time/3600.,asic2,label='ASIC2')
ax2.set_xlabel('Time (hour)')
ax2.set_ylabel('Temperature (C$^\cdot$)')
ax2.legend()
ax2.minorticks_on()
ax3 = plt.subplot(gs[3])
ax3.scatter(voltage*1000.,asic0,s=5,label='ASIC0')
ax3.scatter(voltage*1000.,asic2,s=5,label='ASIC2')
ax3.set_xlabel('Voltage (mV)')
ax3.set_ylabel('Temperature (C$^\cdot$)')
ax3.legend()
ax3.minorticks_on()
plt.tight_layout()
plt.savefig('/Users/mileswinter/Documents/UW/Research/CTA/Presentations/2017_08_02/tri_plot.pdf')
plt.show()
