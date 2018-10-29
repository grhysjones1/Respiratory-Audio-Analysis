#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:48:12 2018

@author: garethjones
"""

#%%

# Explore signal data
import wave

sample1 = wave.open("/users/garethjones/Documents/Data Science/Feebris/Data/Initial Samples/sample1.wav",'r')
nchannels, sampwidth, sampfreq, nframes, comptype, compname = sample1.getparams()

print("Number of channels = "+str(nchannels))
print("Sample width in bytes = "+str(sampwidth))
print("Sampling frequency = "+str(sampfreq))
print("Number of audio frames = "+str(nframes))
print("Compression type = "+str(compname))


#%%
'''
# Listen to signal
import pyaudio 
sample1 = wave.open("/users/garethjones/Documents/Data Science/Feebris/Data/sample1.wav",'r')
chunk = 512  

#instantiate PyAudio and read data 
p = pyaudio.PyAudio()
stream = p.open(format = p.get_format_from_width(sampwidth),channels = nchannels,rate = sample1.getframerate(),output = True)
data = sample1.readframes(chunk)

#play stream  
while data:  
    stream.write(data)
    data = sample1.readframes(chunk)

#stop stream  
stream.stop_stream()  
stream.close()  
p.terminate()  
'''

#%%

# Write signal data to variables

from scipy.io import wavfile
import wave
import numpy as np

def get_wav(file_name, nsamples):
    wav = wavfile.read(file_name)[1]
    signal = wav[0:nsamples]
    return signal

filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Initial Samples/"

# work out length of each sample in frames
sampleframes = []
for i in range(1,4):
    sample = wave.open(filepath+"sample{}.wav".format(i),'r')
    nframes = sample.getnframes()
    sampleframes.append(nframes)
    
# import signals and store
signals = []
for i in range(3):
    signal = get_wav(filepath+"sample{}.wav".format(i+1),sampleframes[i]*2)
    signals.append(signal)
    
# create list of second intervals for each sample
samplesecs = []
for i in range(3):
    xticks = np.arange(1,len(signals[i])+1)/44100
    samplesecs.append(xticks)
    

#%%
    
# normalize signals

# we see each wave file is int16 type, meaining it can be integer value from -2^15 to +2^15
# print(get_wav(filepath+"sample1.wav").dtype)

# convert sample to floating point array between -1 to 1
for i in range(3):
    signals[i] = signals[i] / (2.**15)
    
    
#%%

# Plot wave file
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, (ax1, ax2, ax3) = plt.subplots(3,1)
fig.set_figheight(5)
fig.set_figwidth(8)
plt.subplots_adjust(hspace=1)

ax1.plot(samplesecs[0],signals[0],color='r')
ax1.set_title('Signal 1',pad=12)
ax1.set_xlim(xmin=0)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax2.plot(samplesecs[1],signals[1],color='b')
ax2.set_title('Signal 2',pad=15)
ax2.set_xlim(xmin=0)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax3.plot(samplesecs[2],signals[2])
ax3.set_title('Signal 3',pad=15)
ax3.set_xlim(xmin=0)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

