#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:18:00 2018

@author: garethjones
"""

#%%

''' IMPORTS AND PARAMETERS '''
from scipy.signal import get_window, butter, lfilter
import matplotlib.pyplot as plt

sampfreq = 44_100
nyq_rate = sampfreq/2
windowspersec = 1000/20 # window is every 20 millisecs


#%%

''' CREATE WINDOW '''
windowsize = int(sampfreq/windowspersec)
window = get_window('hamming',windowsize)
plt.plot(window)


#%%

''' DEFINE FILTER AND GET COEFFS '''
def butter_lowpass(cutoff, nyq_rate, order=6):
    normal_cutoff = cutoff / nyq_rate
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_highpass(cutoff, nyq_rate, order=6):
    normal_cutoff = cutoff / nyq_rate
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

b, a = butter_highpass(140,nyq_rate)


#%%
# change window so that there is overlap against the last one

''' APPLY FILTER TO WINDOWED SIGNAL '''
# work with first channel
signal_filtered = np.empty(0)
for i in range(int(signal.shape[1]/windowsize)):
    if i == 0:
        windowed = signal[0,:windowsize] * window
        filtered = lfilter(b, a, windowed)
        signal_filtered = np.append(signal_filtered,filtered)
    else:
        windowed = signal[0,i*windowsize:(i+1)*windowsize] * window
        filtered = lfilter(b, a, windowed)
        signal_filtered = np.append(signal_filtered,filtered)

signal_filtered = np.reshape(signal_filtered,(1,len(signal_filtered))).astype(signal.dtype)


#%%

''' OUTPUT SIGNAL TO AUDIO '''
filepath = '/users/garethjones/Documents/Data Science/Feebris/Data/Initial Samples/'
outname = 'filtered1.wav'

wav_file = wave.open(filepath+outname, "w")
wav_file.setparams((1, sampwidth, sampfreq, 362502, comptype, compname))
wav_file.writeframes(signal_filtered.tobytes('C'))
wav_file.close()
