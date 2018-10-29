#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:42:00 2018

@author: garethjones
"""

#%%
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
import warnings
warnings.filterwarnings('ignore')
samprate = 44100

#%%
# select only one channel of stereo signal, and transpose ready for melspectogram
signals_mono = [signals[i].T[0] for i in range(len(signals))]

# create the mel spectogram power 
mel = [melspectrogram(signals_mono[i],sr=samprate) for i in range(len(signals_mono))]

# convert the power spectogram to dB
mel_db = [librosa.power_to_db(mel[i],ref=np.max) for i in range(len(mel))]


#%%

# plot the mel spectograms of our samples
plt.figure(figsize=(8.5,8))

for i in range(len(mel_db)):
    ax = plt.subplot(3,1,i+1)
    specshow(mel_db[i],y_axis='hz',x_axis='time',sr=samprate)
    plt.ylim((0,5000))
    plt.title('Sample {}'.format(i+1))
    plt.tight_layout()

plt.show()
plt.close()