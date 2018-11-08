#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:10:46 2018

@author: garethjones
"""

#%%

''' IMPORT AUDIO SIGNALS '''

# Write signal data to variables
from scipy.io import wavfile
import wave
import numpy as np
samprate = 44100

def get_wav(file_name, nsamples):
    wav = wavfile.read(file_name)[1]
    signal = wav[0:nsamples]
    return signal

filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"

# work out length of each sample in frames
sampleframes = []
for i in range(1,11):
    sample = wave.open(filepath+"Signals/Signal {}.wav".format(i),'r')
    nframes = sample.getnframes()
    sampleframes.append(nframes)
    
# import signals and store
signals = [get_wav(filepath+"Signals/Signal {}.wav".format(i+1),sampleframes[i]*2) for i in range(10)]
    
# create list of second intervals for each sample
samplesecs = [np.arange(1,len(signals[i])+1) / samprate for i in range(10)]
    


''' IMPORT ANNOTATION AUDIO SIGNALS '''

# work out length of each sample in frames
annotationframes = []
for i in range(1,11):
    sample = wave.open(filepath+"Annotations/Annotation {}.wav".format(i),'r')
    nframes = sample.getnframes()
    annotationframes.append(nframes)
    
# import signals and store
annotations = [get_wav(filepath+"Annotations/Annotation {}.wav".format(i+1),annotationframes[i]*2) for i in range(10)]



''' NORMALIZE SIGNALS TO 1 '''

# Normalize signals
for i in range(len(signals)):
    signals[i] = signals[i] / (2.**15)

for i in range(len(annotations)):
    annotations[i] = annotations[i] / (2.**15)    



''' CUT SIGNALS TO SAME LENGTH '''

# 2,650,000 is the right length based on eye-balling the data
# need to automate this
for i in range(len(signals)):
    signals[i] = signals[i][0:2650000]

for i in range(len(signals)):
    annotations[i] = annotations[i][0:2650000]

# test to ensure lengths are the same
for i in range(len(signals)):
    assert len(signals[i]) == len(annotations[i])



''' MAKE SIGNALS & ANNOTATIONS MONO '''

# select only one channel of stereo signal, and transpose ready for melspectogram
signals_mono = [signals[i].T[0] for i in range(len(signals))]

# make annotations one channel, transposed, absolute
annotations_mono = [abs(annotations[i].T[0]) for i in range(len(annotations))]



''' VISUALISE '''

import matplotlib.pyplot as plt

signal_num = 4  # set which signal you want to see

fig, axs = plt.subplots(2,1,figsize=(10,6))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Input Signal & Annotation Audio #{}'.format(signal_num),weight='bold')

axs[0].plot(signals_mono[signal_num])
axs[0].set_xlim(xmin=0,xmax=len(signals_mono[signal_num]))
axs[0].set_title('Signal {}'.format(signal_num),pad=10)
axs[0].spines['top'].set_color('none')
axs[0].spines['right'].set_color('none')

axs[1].plot(annotations_mono[signal_num])
axs[1].set_xlim(xmin=0,xmax=len(annotations_mono[signal_num]))
axs[1].set_title('Annotations {}'.format(signal_num),pad=10)
axs[1].spines['top'].set_color('none')
axs[1].spines['right'].set_color('none')

plt.show()
plt.close()



''' GLOBAL VARIABLES FOR REFERENCE '''

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



''' TURN ANNOTATIONS INTO BINARY SIGNAL '''

# denote whenever amplitude is above threshold
anno_gates = []
for i in range(len(annotations_mono)):
    gate_list = []
    for j in range(len(annotations_mono[i])):
        if annotations_mono[i][j] > thresholds['Annotation {}'.format(i+1)]: # this is amplitude threshold
            x = 1
        else:
            x = 0
        gate_list.append(x)
    anno_gates.append(gate_list)



''' SUPRESS NOISE IN BINARY ANNOTATIONS '''

# ensure noise is removed so there's exact number of clicks
fwd_frame_thresh = 15000

size_anno_gates = []
for i in range(len(anno_gates)):
    for j in range(len(anno_gates[i])):
        if anno_gates[i][j] == 1:
            for k in range(1,fwd_frame_thresh): # this is the forward threshold for silencing frames
                if j+k < len(anno_gates[i]):
                    anno_gates[i][j+k] = 0
                else:
                    k = fwd_frame_thresh - j
                    anno_gates[i][j+k] = 0       
    size_anno_gates.append(sum(anno_gates[i]))

print(np.r_[list(newclickspersignal.values())] - np.r_[size_anno_gates])



''' VISUALISE ANNOTATION MONO & ANNOTATION BINARY '''

signal_num = 1

fig, axs = plt.subplots(2,1,figsize=(10,6))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Annotations for Signal #{}'.format(signal_num),weight='bold')

axs[0].plot(annotations_mono[signal_num])
axs[0].set_title('Original Annotation Audio Signal (Mono)',pad=10)
axs[0].set_xlim(xmin=0,xmax=len(annotations_mono[signal_num]))
axs[0].set_ylim(ymin=0)
axs[0].spines['top'].set_color('none')
axs[0].spines['right'].set_color('none')

axs[1].plot(anno_gates[signal_num])
axs[1].set_title('Binary Annotation Signal No Noise',pad=10)
axs[1].set_xlim(xmin=0,xmax=len(anno_gates[signal_num]))
axs[1].set_ylim(ymin=0)
axs[1].spines['top'].set_color('none')
axs[1].spines['right'].set_color('none')

plt.show()
plt.close()