#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:12:11 2018

@author: garethjones
"""

#%%
''' IMPORTS AND PARAMETER SETTING ''' 
# file imports
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np
import wave
import math

# set file names    
filepath = '/users/garethjones/Documents/Data Science/Feebris/Data/Initial Samples/'
filename = 'sample1.wav'
outname = 'filtered1.wav'

# get parameters of input signal
sample1 = wave.open(filepath+filename,'r')
nchannels, sampwidth, sampfreq, nframes, comptype, compname = sample1.getparams()
sample1.close()
nyq_rate = 0.5 * sampfreq


''' GET RAW AUDIO '''
# define function to get raw audio data
def get_wav(file_name, nsamples=nframes*2):
    wav = wavfile.read(file_name)[1]
    signal = wav[0:nsamples]
    return signal

# get raw audio data
signal = get_wav(filepath+filename)
signal = signal.T


''' DEFINE AND RUN FILTER '''
# define function for filter design
# running mean is computing the average of the signals within the window size
def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

# define filter frequency
cutOffFrequency = 100

# calculated window size
freqRatio = (cutOffFrequency/sampfreq)
N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

# Use moviung average filter (only on first channel)
filtered = running_mean(signal[0], N).astype(signal.dtype)


''' OUTPUT AUDIO FILE '''
wav_file = wave.open(filepath+outname, "w")
wav_file.setparams((1, sampwidth, sampfreq, nframes, comptype, compname))
wav_file.writeframes(filtered.tobytes('C'))
wav_file.close()
