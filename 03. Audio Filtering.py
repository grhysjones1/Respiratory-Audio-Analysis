#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:48:12 2018

@author: garethjones
"""

#%%

# Define filter types
import numpy as np
import scipy.signal as sig
from scipy.signal import butter, lfilter, freqz, lp2bp, lp2hp

def butter_lowpass(cutoff, nyq_rate, order=6):
    normal_cutoff = cutoff / nyq_rate
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_highpass(cutoff, nyq_rate, order=6):
    normal_cutoff = cutoff / nyq_rate
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, nyq_rate, order=6):
    normal_lowcut = lowcut / nyq_rate
    normal_highcut = highcut / nyq_rate
    Wn = [normal_lowcut,normal_highcut]
    b, a = butter(order, Wn, btype='bandpass', analog=False)
    return b, a


#%%

# Plot freq response of filters
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style('chesterish')

# Filter inputs
samp_rate = 44100
nyq_rate = 0.5 * samp_rate
lowcut = 10000
highcut = 8000
b_lowcut = 3000
b_highcut = 10000
lfilt_order = 6
hfilt_order = 6
bfilt_order = 6

# Get filter coefficients
b_low, a_low = butter_lowpass(lowcut, nyq_rate, order=lfilt_order)
b_high, a_high = butter_highpass(highcut, nyq_rate, hfilt_order)
b_band, a_band = butter_bandpass(b_lowcut, b_highcut, nyq_rate, bfilt_order)

# Get the frequency response.
w_low, h_low = freqz(b_low, a_low)
w_high, h_high = freqz(b_high, a_high)
w_band, h_band = freqz(b_band, a_band)


#%%

# Plot shape of filter
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
fig.set_figheight(11)
fig.set_figwidth(11)
plt.subplots_adjust(hspace=0.6)

ax1.plot(nyq_rate*w_low/np.pi, np.abs(h_low), 'b')
ax1.plot(lowcut, 0.5*np.sqrt(2), 'ko')
ax1.axvline(lowcut, color='r')
ax1.set_title('Low Pass Filter Response',pad=12)
ax1.set_xlim(0, nyq_rate)

ax2.plot(nyq_rate*w_high/np.pi, np.abs(h_high), 'b')
ax2.plot(highcut, 0.5*np.sqrt(2), 'ko')
ax2.axvline(highcut, color='r')
ax2.set_title('High Pass Filter Response',pad=12)
ax2.set_xlim(0, nyq_rate)

ax3.plot(nyq_rate*w_band/np.pi, np.abs(h_band), 'b')
ax3.plot(b_lowcut, 0.5*np.sqrt(2), 'ko')
ax3.axvline(b_lowcut, color='r')
ax3.plot(b_highcut, 0.5*np.sqrt(2), 'ko')
ax3.axvline(b_highcut, color='r')
ax3.set_title('Band Pass Filter Response',pad=12)
ax3.set_xlim(0, nyq_rate)

plt.show()
plt.close()


#%%
# THESE FILTERS DON'T SEEM TO BE WORKING!

# get filter coefficients
b_low, a_low = butter_lowpass(2000, nyq_rate, order=3)
b_high, a_high = butter_highpass(1000, nyq_rate, order=2)
b_band, a_band = butter_bandpass(3000, 12000, nyq_rate, order=1)

# Filter signals
wav_lfilter = lfilter(b_low, a_low, signals[0])
wav_hfilter = lfilter(b_high, a_high, signals[0])
wav_bpfilter = lfilter(b_band, a_band, signals[0])

# Write to wav files
wavfile.write(filepath+'signal1_lowpassed.wav',sampfreq,wav_lfilter)
wavfile.write(filepath+'signal1_highpassed.wav',sampfreq,wav_hfilter)
wavfile.write(filepath+'signal1_bandpassed.wav',sampfreq,wav_bpfilter)


#%%

# Plot filtered against original
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
fig.set_figheight(7)
fig.set_figwidth(13)
plt.subplots_adjust(hspace=1)

ax1.plot(samplesecs[0],signals[0],color='r')
ax1.set_title('Original Signal 1',pad=12)
ax1.set_xlim(xmin=0)
ax1.set_ylim(bottom=-1,top=1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax2.plot(samplesecs[0],wav_lfilter,color='r')
ax2.set_title('Low Filtered Signal 1',pad=12)
ax2.set_xlim(xmin=0)
ax2.set_ylim(bottom=-1,top=1)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax3.plot(samplesecs[0],wav_hfilter,color='r')
ax3.set_title('High Filtered Signal 1',pad=12)
ax3.set_xlim(xmin=0)
ax3.set_ylim(bottom=-1,top=1)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax4.plot(samplesecs[0],wav_bpfilter,color='r')
ax4.set_title('Bandpassed Filtered Signal 1',pad=12)
ax4.set_xlim(xmin=0)
ax4.set_ylim(bottom=-1,top=1)
ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.5))


#%%
# See step response of filter

step_response = lfilter(b_low,a_low,np.ones(50))
plt.plot(step_response)
plt.show()
plt.close()

step_response = lfilter(b_high,a_high,np.ones(50))
plt.plot(step_response)
plt.show()
plt.close()










