#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:28:49 2018

@author: garethjones
"""

#%%

''' IMPORTS & GLOBALS '''

import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
import pylab
import warnings
warnings.filterwarnings('ignore')
samprate = 44100


#%%

''' GENERATE MEL SPECTOGRAMS FOR SIGNALS '''

# select only one channel of stereo signal, and transpose ready for melspectogram
signals_mono = [signals[i].T[0] for i in range(len(signals))]

# create the mel spectogram power 
# hop_length = 512 (number of frames skipped until next window). this gives one bucket every 0.13s, which could be changed
# n_fft = 1024 (number of frames either side to calculate the FFT)
mel = [melspectrogram(signals_mono[i],sr=samprate,n_fft=1024) for i in range(len(signals_mono))]

# convert the power spectogram to dB
mel_db = [librosa.power_to_db(mel[i],ref=np.max) for i in range(len(mel))]


#%%

''' SAVE MELSPECTOGRAMS TO FILE '''

for i in range(len(mel_db)):
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    specshow(mel_db[i],y_axis='hz',x_axis='time',sr=samprate)
    pylab.ylim(0,3000)
    pylab.savefig(filepath+'Spectograms/Signal {}.png'.format(i+1),bbox_inches=None,pad_inches=0,dpi=1000)
    pylab.close()


#%%

''' VISUALISE '''

plt.figure(figsize =(8,4))
specshow(mel_db[1],y_axis='hz',x_axis='time',sr=samprate)
plt.ylim((0,3000))
plt.axis('off')
plt.margins(0)
plt.tight_layout()
plt.close()