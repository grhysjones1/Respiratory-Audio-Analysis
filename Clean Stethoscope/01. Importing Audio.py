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
    

#%%

''' IMPORT ANNOTATION AUDIO SIGNALS '''

# work out length of each sample in frames
annotationframes = []
for i in range(1,11):
    sample = wave.open(filepath+"Annotations/Annotation {}.wav".format(i),'r')
    nframes = sample.getnframes()
    annotationframes.append(nframes)
    
# import signals and store
annotations = [get_wav(filepath+"Annotations/Annotation {}.wav".format(i+1),annotationframes[i]*2) for i in range(10)]


#%%

''' NORMALIZE SIGNALS TO 1 '''

# Normalize signals
for i in range(len(signals)):
    signals[i] = signals[i] / (2.**15)

for i in range(len(annotations)):
    annotations[i] = annotations[i] / (2.**15)    


#%%

''' CUT OUT SIGNALS WHERE ANNOTATIONS ARE NOT CLEAR '''

# set all signals to be the length of annotations
for i in range(len(signals)):
    signals[i] = signals[i][0:len(annotations[i])]

# pre-process signals 1,2,7 specifically due to poor click annotations
signals[0] = signals[0][0:2700000]
annotations[0] = annotations[0][0:2700000]

signals[1] = signals[1][0:2650000]
annotations[1] = annotations[1][0:2650000]

signals[6] = signals[6][0:2800000]
annotations[6] = annotations[6][0:2800000]

# test to ensure lengths are the same
for i in range(len(signals)):
    assert len(signals[i]) == len(annotations[i])

#%%

''' CUT SIGNALS TO SHORTEST LENGTH (FOR EQUAL SIZED SPECTOGRAMS) '''

# to ensure spectograms work well, I need to cut the data at 2,650,000, the shortest sample length, so everything is same length
# get all signal lengths
signallen = [len(signals[i]) for i in range(len(signals))] 

for i in range(len(signals)):
    signals[i] = signals[i][0:min(signallen)]

for i in range(len(signals)):
    annotations[i] = annotations[i][0:min(signallen)]


#%%

''' VISUALISE '''

import matplotlib.ticker as ticker
fig,ax = plt.subplots(1,1,figsize=(20,8))
ax.plot(annotations[6])
ax.xaxis.set_major_locator(ticker.MultipleLocator(125000))








