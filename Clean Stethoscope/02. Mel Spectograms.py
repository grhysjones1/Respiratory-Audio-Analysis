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

''' MAKE SIGNALS MONO AUDIO '''

# select only one channel of stereo signal, and transpose ready for melspectogram
signals_mono = [signals[i].T[0] for i in range(len(signals))]


#%%

''' GENERATE MEL SPECTOGRAMS FOR SIGNALS '''

hop_length = 2550  # number of frames to jump when computing fft
fmin = 125  # bottom frequency to look at
fmax = 500  # top frequency to look at
n_mels = 45  # number of audio frequency bins
n_fft = [8000, 11025, 14000]  # width of the fft windows


# list of width 3 different mels per length 10 signals
mel_db_grid = []

for i in range(len(signals_mono)):
    
    mel = [melspectrogram(
            signals_mono[i],
            sr = samprate,
            hop_length = 2550,
            n_fft = j,
            n_mels = 45,
            fmin = 125,
            fmax = 500) for j in n_fft]
    
    mel_db = [librosa.power_to_db(mel[k],ref=np.max) for k in range(len(mel))]
    
    mel_db_grid.append(mel_db)


#%%
# find shape of mels
print(mel_db_grid[0][2].shape)










#%%

''' JUST USE FOR VISUALISATION, NOT INPUT TO MODEL '''
''' SAVE MELSPECTOGRAMS TO FILE '''

for i in range(len(mel_db)):
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    specshow(mel_db[i],y_axis='hz',x_axis='time',sr=samprate)
    pylab.savefig(filepath+'Spectograms/Signal {}.png'.format(i+1),bbox_inches=None,pad_inches=0,dpi=1000)
    pylab.close()


#%%

''' FIND GOOD SPECTOGRAM VALUES '''

# 1s = 44,100    0.5s = 22,050   0.25s = 11,025   0.125s = 5513    0.0625s = 2756

#hop_frames = [1250,2550,5500,11000,22050]
hop_frames = [256,512,1250,2550,5500]
fft_width = [8000, 11025, 14000]
fmin = 125
fmax = 500
freq_bins = [30, 35, 40, 45]
signal = signals_mono[2]

for i in fft_width:
    for j in freq_bins:    
        plt.figure(figsize=(15,15))
        for k in range(1,6):
            meltest = melspectrogram(signal,samprate,hop_length=hop_frames[k-1],n_fft=i,fmin=fmin,fmax=fmax,n_mels=j)
            meltest_db = librosa.power_to_db(meltest,ref=np.max)
            plt.subplot(5,1,k)
            plt.title('Hop Frames = {}'.format(hop_frames[k-1]))
            specshow(meltest_db,sr=samprate)
        plt.suptitle('FFT Window Width = {} / Number Frequency Bin = {}'.format(i,j))
        plt.savefig(filepath+'Comparison FFTWindow {} Freqbins {}.png'.format(i,j),dpi=900)
        plt.show()
        plt.close()


#%%
plt.figure(figsize=(15,15))
for k in range(1,9):
    meltest = melspectrogram(signal,samprate,hop_length=hop_frames[k-1],n_fft=11025,fmin=fmin,fmax=fmax,n_mels=40)
    meltest_db = librosa.power_to_db(meltest,ref=np.max)
    plt.subplot(5,1,k)
    plt.title('Hop Frames = {}'.format(hop_frames[k-1]))
    specshow(meltest_db,sr=samprate)
plt.suptitle('FFT Window Width = 11025 / Number Frequency Bin = 40'.format(i,j))
plt.savefig(filepath+'Comparison FFTWindow 11025 Freqbins 40.png'.format(i,j),dpi=900)
plt.show()
plt.close()

  
#%%

plt.figure(figsize=(17,25))

meltest = melspectrogram(signal,samprate,hop_length=2550,n_fft=8000,fmin=fmin,fmax=fmax,n_mels=30)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,1)
plt.title('FFT 8000 Freqs 30 Hop 2550 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=2550,n_fft=8000,fmin=fmin,fmax=fmax,n_mels=35)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,2)
plt.title('FFT 8000 Freqs 35 Hop 2550 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=2550,n_fft=8000,fmin=fmin,fmax=fmax,n_mels=40)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,3)
plt.title('FFT 8000 Freqs 40 Hop 2550 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=2550,n_fft=8000,fmin=fmin,fmax=fmax,n_mels=45)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,4)
plt.title('FFT 8000 Freqs 45 Hop 2550 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=1250,n_fft=11025,fmin=fmin,fmax=fmax,n_mels=35)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,5)
plt.title('FFT 11025 Freqs 35 Hop 1250 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=1250,n_fft=11025,fmin=fmin,fmax=fmax,n_mels=40)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,6)
plt.title('FFT 11025 Freqs 40 Hop 1250 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=2550,n_fft=11025,fmin=fmin,fmax=fmax,n_mels=45)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,7)
plt.title('FFT 11025 Freqs 45 Hop 2550 ')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signal,samprate,hop_length=5500,n_fft=14000,fmin=fmin,fmax=fmax,n_mels=40)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(8,1,8)
plt.title('FFT 14000 Freqs 40 Hop 5500 ')
specshow(meltest_db,sr=samprate)

plt.show()
plt.close()


#%%

''' LOOK AT MELS OF 3 DIFFERENT FFT WINDOWS, SAME HOP AND NMELS '''

fmin = 125
fmax = 500
signalnum = 3

plt.figure(figsize=(10,15))

meltest = melspectrogram(signals_mono[signalnum],samprate,hop_length=2550,n_fft=8000,fmin=fmin,fmax=fmax,n_mels=45)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(3,1,1)
plt.title('FFT 8000 Freqs 45 Hop 2550')
specshow(meltest_db,sr=samprate,x_axis='frames')

meltest = melspectrogram(signals_mono[signalnum],samprate,hop_length=2550,n_fft=11025,fmin=fmin,fmax=fmax,n_mels=45)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(3,1,2)
plt.title('FFT 11025 Freqs 45 Hop 2550')
specshow(meltest_db,sr=samprate)

meltest = melspectrogram(signals_mono[signalnum],samprate,hop_length=2550,n_fft=14000,fmin=fmin,fmax=fmax,n_mels=45)
meltest_db = librosa.power_to_db(meltest,ref=np.max)
plt.subplot(3,1,3)
plt.title('FFT 14000 Freqs 45 Hop 2550')
specshow(meltest_db,sr=samprate)
