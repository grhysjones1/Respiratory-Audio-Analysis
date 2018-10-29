#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 22:51:49 2018

@author: garethjones
"""

#%%

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq, ifft



n = 1000 # number of samples
Lx = 100 # time of our sample
omg = 2*np.pi/Lx 

x = np.linspace(0,Lx,n)
y1 = np.cos(5.0*omg*x)
y2 = np.sin(10.0*omg*x)
y3 = np.sin(20*omg*x)

y = y1+y2+y3

plt.plot(x,y)
plt.show()
plt.close()

freqs = fftfreq(n)
mask = freqs > 0
fft_vals = fft(y)

# theoretical fft - this is getting rid of the complex conjugates
fft_theo = 2.0*np.abs(fft_vals/n)

# plt.plot(freqs,fft_vals)
plt.plot(freqs[mask],fft_theo[mask])
plt.show()  
plt.close()


# try changing the length to be one - the resolution should be number of samples second

freqs = fftfreq(len(signals[0]))
mask = freqs > 0 # discards negative frequencies
fft_vals = fft(signals[0]) # this is dependant on sampling frequency
fft_theo = 2*np.abs(fft_vals / len(signals[0]))
# plt.plot(freqs,fft_vals)
plt.plot(freqs[mask],fft_theo[mask])
plt.show()
plt.close()

fft_theo[(freqs>0.4)] = 0
plt.plot(freqs[mask],fft_vals[mask])
plt.show()
plt.close()

cut_signal = ifft(fft_theo)

fig, (ax1) = plt.subplots(1,1)
fig.set_figheight(5)
fig.set_figwidth(8)
plt.subplots_adjust(hspace=1)

ax1.plot(samplesecs[0],cut_signal,color='r')
ax1.set_title('Signal 1',pad=12)
ax1.set_xlim(xmin=0)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))



