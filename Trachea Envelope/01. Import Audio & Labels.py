#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:00:02 2018

@author: garethjones
"""

# Write signal data to variables

import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter, butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from scipy.io import wavfile
import numpy as np
import librosa
import wave

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

filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Trachea Envelope/"

# get chest audio signals
chest_mono = [get_signals(filepath+"P1_Signal_{}.wav".format(i+1),2600000) for i in range(8)]
    
# get trachea audio signls
trachea_mono = [get_signals(filepath+"P1_Signal_{}_T.wav".format(i+1),2600000) for i in range(8)]


#%%
# Plot wave files
fig, axs = plt.subplots(8,1)
fig.set_figheight(15)
fig.set_figwidth(8)
plt.subplots_adjust(hspace=1)
for i in range(len(chest_mono)):
    axs[i].plot(chest_mono[i],color='r')
    axs[i].set_title('Chest Signal {}'.format(i+1),pad=12)
    axs[i].set_xlim(xmin=0)

fig, axs = plt.subplots(8,1)
fig.set_figheight(15)
fig.set_figwidth(8)
plt.subplots_adjust(hspace=1)
for i in range(len(trachea_mono)):
    axs[i].plot(trachea_mono[i],color='b')
    axs[i].set_title('Trachea Signal {}'.format(i+1),pad=12)
    axs[i].set_xlim(xmin=0)



#%%

# get bottom mfcc bands of trachea signals
trachea_mfcc = [librosa.feature.mfcc(trachea_mono[i], sr=44100, n_mfcc=12,hop_length=441,fmin=125,fmax=2000)[1] for i in range(len(trachea_mono))]

# standardize mfcc band signal
trachea_mfcc_normed = []
for i in range(8):
    mean = trachea_mfcc[i].mean()
    std = trachea_mfcc[i].std()
    normed = (trachea_mfcc[i] - mean)/std
    trachea_mfcc_normed.append(normed)

# set labelling thresholds
mfcc_thresh = 0
fwd_thresh = 20

mfcc_thresholds_normed = []

for i in range(len(trachea_mfcc_normed)):
    
    env_thresh = np.zeros(len(trachea_mfcc_normed[i]))
    
    for j in range(len(trachea_mfcc_normed[i])):
        if trachea_mfcc_normed[i][j] > mfcc_thresh:
            env_thresh[j] = 1
        else:
            env_thresh[j] = 0
    
    indices_up = []
    for i in range(len(env_thresh)-1):
        if env_thresh[i] == 0:
            if env_thresh[i+1] == 1:
                x = i+1
                indices_up.append(x)
    
    for i in indices_up:
        if min(env_thresh[i:i+fwd_thresh]) == 0:
            env_thresh[i:i+fwd_thresh] = 0
    
    indices_down = []
    for i in range(len(env_thresh)-1):
        if env_thresh[i] == 1:
            if env_thresh[i+1] == 0:
                x = i+1
                indices_down.append(x)
    
    for i in indices_down:
        if max(env_thresh[i:i+fwd_thresh]) == 1:
            env_thresh[i:i+fwd_thresh] = 1

    mfcc_thresholds_normed.append(env_thresh)

# vistualise to ensure labels are roughly correct
fig, axs = plt.subplots(8,1)
fig.set_figheight(30)
fig.set_figwidth(15)
plt.subplots_adjust(hspace=0.6)
for i in range(len(trachea_mfcc)):
    axs[i].plot(trachea_mfcc_normed[i],color='g')
    axs[i].plot(mfcc_thresholds_normed[i],color='b')
    axs[i].set_title('Trachea MFCC1 Signal {}'.format(i+1),pad=12)
    axs[i].set_xlim(xmin=0,xmax=len(trachea_mfcc_normed[i]))


#%%
    
# get mfccs of chest signals
chest_mfcc = [librosa.feature.mfcc(chest_mono[i], sr=44100, n_mfcc=10,hop_length=441,n_fft=5000,fmin=125,fmax=2000) for i in range(len(chest_mono))]


# get all trachea mfccs
trachea_mfcc = [librosa.feature.mfcc(trachea_mono[i], sr=44100, n_mfcc=10,hop_length=441,n_fft=5000,fmin=125,fmax=2000) for i in range(len(trachea_mono))]




#%%

''' CHEST MFCC 70:30 WITH RANDOM FOREST '''

# choose 1 signal
signum = 0
signal = chest_mfcc[signum].T
signal = (signal-signal.mean())/signal.std()
labels = mfcc_thresholds_normed[signum]

''' PREP DATA 70/30 SPLIT '''

train_mfcc = signal[0:int(0.7*signal.shape[0]),:]
train_labels = labels[0:int(0.7*len(labels))] 

test_mfcc = signal[int(0.7*signal.shape[0]):,:]
test_labels = labels[int(0.7*len(labels)):]

assert train_mfcc.shape[0] == len(train_labels)
assert test_mfcc.shape[0] == len(test_labels)

''' RANDOM FOREST FOR 70:30 CHUNK '''

rfc = RandomForestClassifier(n_estimators=200, random_state=123)
rfc.fit(train_mfcc,train_labels)
acc = rfc.score(test_mfcc,test_labels)

print('Model Accuracy Chest Signal = {:.1%}'.format(acc))



#%%

''' CHEST SIGNAL LEAVE ONE OUT APPROACH '''

train_mfcc = np.hstack(chest_mfcc[:-1]).T
train_labels = np.hstack(mfcc_thresholds_normed[:-1])

test_mfcc = chest_mfcc[-1].T
test_labels = mfcc_thresholds_normed[-1]

assert train_mfcc.shape[0] == len(train_labels)
assert test_mfcc.shape[0] == len(test_labels)


rfc = RandomForestClassifier(n_estimators=200, random_state=123)
rfc.fit(train_mfcc,train_labels)
acc = rfc.score(test_mfcc,test_labels)

print('Model Accuracy = {:.1%}'.format(acc))



#%%

''' SINGLE TRACHEA MFCC 70:30 WITH RANDOM FOREST '''

# choose 1 signal
signum = 0
signal = trachea_mfcc[signum].T
signal = (signal-signal.mean())/signal.std()
labels = mfcc_thresholds_normed[signum]

''' PREP DATA 70/30 SPLIT '''

train_mfcc = signal[0:int(0.7*signal.shape[0]),:]
train_labels = labels[0:int(0.7*len(labels))] 

test_mfcc = signal[int(0.7*signal.shape[0]):,:]
test_labels = labels[int(0.7*len(labels)):]

assert train_mfcc.shape[0] == len(train_labels)
assert test_mfcc.shape[0] == len(test_labels)

rfc = RandomForestClassifier(n_estimators=200, random_state=123)
rfc.fit(train_mfcc,train_labels)
acc = rfc.score(test_mfcc,test_labels)

print('Model Accuracy Trachea Signal = {:.1%}'.format(acc))


#%%

''' ALL TRACHEA MFCC 70:30 WITH RANDOM FOREST '''

trachea_mfcc_all_normed = []
for i in range(len(trachea_mfcc)):
    signal = trachea_mfcc[i].T
    signal = (signal-signal.mean())/signal.std()
    trachea_mfcc_all_normed.append(signal)

length = trachea_mfcc_all_normed[0].shape[0]

train_mfcc = [trachea_mfcc_all_normed[i][0:int(0.7*length),:] for i in range(len(trachea_mfcc_all_normed))]
train_mfcc = np.vstack(train_mfcc)
train_labels = [mfcc_thresholds_normed[i][0:int(0.7*length)] for i in range(len(mfcc_thresholds_normed))] 
train_labels = np.hstack(train_labels)

test_mfcc = [trachea_mfcc_all_normed[i][int(0.7*length):,:] for i in range(len(trachea_mfcc_all_normed))]
test_mfcc = np.vstack(test_mfcc)
test_labels = [mfcc_thresholds_normed[i][int(0.7*length):] for i in range(len(mfcc_thresholds_normed))] 
test_labels = np.hstack(test_labels)

rfc = RandomForestClassifier(n_estimators=100, random_state=123)
rfc.fit(train_mfcc,train_labels)
acc = rfc.score(test_mfcc,test_labels)

print('Model Accuracy = {:.1%}'.format(acc))


#%%

''' LEAVE ONE OUT APPROACH TRACHEA '''

train_mfcc = np.hstack(trachea_mfcc[:-1]).T
train_labels = np.hstack(mfcc_thresholds_normed[:-1])

test_mfcc = trachea_mfcc[-1].T
test_labels = mfcc_thresholds_normed[-1]

assert train_mfcc.shape[0] == len(train_labels)
assert test_mfcc.shape[0] == len(test_labels)


rfc = RandomForestClassifier(n_estimators=100, random_state=123)
rfc.fit(train_mfcc,train_labels)
acc = rfc.score(test_mfcc,test_labels)

print('Model Accuracy = {:.1%}'.format(acc))