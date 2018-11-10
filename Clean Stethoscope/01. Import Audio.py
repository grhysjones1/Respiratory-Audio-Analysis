#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:10:46 2018

@author: garethjones
"""

#%%

''' DEFINE FUNCTIONS '''

def get_signals(filename,trim_length):
    
    # read in audio file as numpy array
    sample = wave.open(filename,'r')
    nframes = sample.getnframes()
    wav = wavfile.read(filename)[1]
    signal = wav[0:nframes*2]
    
    # normalize
    signal = signal / (2.**15)  # normalize by 16bit width
    
    # trim to given length
    signal = signal[0:trim_length]
    
    # transpose and make mono
    signal_mono = signal.T[0]
    
    return signal_mono


def binarize(signal,ampthresh,fwdthresh):
    
    # create list of 1s and 0s where annotation is above given threshold
    binarized_signal = [1 if signal[i] > ampthresh else 0 for i in range(len(signal))]
    
    # supress noise in binary signal due to anologue sound capture
    for i in range(len(binarized_signal)):
        if binarized_signal[i] == 1: 
            for j in range(1,fwdthresh):
                if i+j < len(binarized_signal):
                    binarized_signal[i+j] = 0
                else:
                    j = fwdthresh - i
                    binarized_signal[i+j] = 0 
    
    return binarized_signal     



#%%

''' GLOBAL VARIABLES '''

# Write signal data to variables
from scipy.io import wavfile
import wave
import numpy as np
samprate = 44100
trim_length = 2650000
fwdthresh = 15000
filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"

# known clicks per sample
originalclickspersignal = {
        'Annotation 1' : 24, 'Annotation 2' : 20, 'Annotation 3' : 24, 'Annotation 4' : 34, 'Annotation 5' : 30,
        'Annotation 6' : 31, 'Annotation 7' : 32, 'Annotation 8' : 36, 'Annotation 9' : 33, 'Annotation 10' : 29}

# these are number of clicks per sample after chopping data
newclickspersignal = {
        'Annotation 1' : 22, 'Annotation 2' : 19, 'Annotation 3' : 22, 'Annotation 4' : 33, 'Annotation 5' : 28,
        'Annotation 6' : 30, 'Annotation 7' : 31, 'Annotation 8' : 29, 'Annotation 9' : 29, 'Annotation 10' : 26}

# eye-balled thresholds for each signal
thresholds = {
        'Annotation 1' : 0.1, 'Annotation 2' : 0.1, 'Annotation 3' : 0.07, 'Annotation 4' : 0.1, 'Annotation 5' : 0.03,
        'Annotation 6' : 0.1, 'Annotation 7' : 0.095, 'Annotation 8' : 0.1, 'Annotation 9' : 0.05, 'Annotation 10' : 0.1}



''' IMPORT AUDIO AND BINARIZE ANNOTATIONS '''

# import respiratory and annotation signals
signals_mono = [get_signals(filepath+"Signals/Signal {}.wav".format(i+1),trim_length) for i in range(10)]
annotations_mono = [get_signals(filepath+"Annotations/Annotation {}.wav".format(i+1),trim_length) for i in range(10)]

# test to ensure lengths are the same
for i in range(len(signals_mono)):
    assert len(signals_mono[i]) == len(annotations_mono[i])

# create binary signals
anno_gates = [binarize(annotations_mono[i],thresholds['Annotation {}'.format(i+1)],fwdthresh) for i in range(len(annotations_mono))]

# find number of annotations in each binary signal
sum_anno_gates = [sum(anno_gates[i]) for i in range(len(anno_gates))]
print(np.r_[list(newclickspersignal.values())] - np.r_[sum_anno_gates])



#%%

''' VISUALISE SIGNAL VS ANNOTATION '''

import matplotlib.pyplot as plt

signal_num = 2  # set which signal you want to see

fig, axs = plt.subplots(3,1,figsize=(10,8))
plt.subplots_adjust(hspace=0.7)
plt.suptitle('Input Signal & Annotation #{}'.format(signal_num),weight='bold')

axs[0].plot(signals_mono[signal_num])
axs[0].set_title('Original Audio Signal {}'.format(signal_num),pad=10)
axs[0].set_xlim(xmin=0,xmax=len(signals_mono[signal_num]))
axs[0].spines['top'].set_color('none')
axs[0].spines['right'].set_color('none')

axs[1].plot(annotations_mono[signal_num])
axs[1].set_title('Original Annotation Audio Signal (Mono)',pad=10)
axs[1].set_xlim(xmin=0,xmax=len(annotations_mono[signal_num]))
axs[2].set_ylim(ymin=0)
axs[1].spines['top'].set_color('none')
axs[1].spines['right'].set_color('none')

axs[2].plot(anno_gates[signal_num])
axs[2].set_title('Binary Annotation Signal No Noise',pad=10)
axs[2].set_xlim(xmin=0,xmax=len(anno_gates[signal_num]))
axs[2].set_ylim(ymin=0)
axs[2].spines['top'].set_color('none')
axs[2].spines['right'].set_color('none')

plt.show()
plt.close()
