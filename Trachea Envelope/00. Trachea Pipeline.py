#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:48:05 2018

@author: garethjones
"""

''' IMPORTS '''

import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter, butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from scipy.io import wavfile
import numpy as np
from librosa.display import specshow
from librosa.feature import melspectrogram
import librosa
import wave
import warnings
warnings.filterwarnings('ignore')


''' IMPORT SIGNALS '''

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


''' GET LABELS FROM TRACHEA MFCC SIGNAL '''

def get_labels(trachea_all,mfcc_thresh=0,fwd_thresh=20,samprate=44100,n_mfcc=12,hop_length=441,fmin=125,fmax=2000):

    # get bottom mfcc bands of trachea signals
    trachea_mfcc_band1 = [librosa.feature.mfcc(trachea_all[i], sr=samprate, n_mfcc=n_mfcc,hop_length=hop_length,fmin=fmin,fmax=fmax)[1] for i in range(len(trachea_all))]
    
    # standardize mfcc band signal
    trachea_mfcc_std = []
    for i in range(len(trachea_all)):
        mean = trachea_mfcc_band1[i].mean()
        std = trachea_mfcc_band1[i].std()
        normed = (trachea_mfcc_band1[i] - mean)/std
        trachea_mfcc_std.append(normed)
    
    # create labels for each frame of signal mfcc
    trachea_labels = []
    for i in range(len(trachea_mfcc_std)):
        
        env_thresh = np.zeros(len(trachea_mfcc_std[i]))
        
        for j in range(len(trachea_mfcc_std[i])):
            if trachea_mfcc_std[i][j] > mfcc_thresh:
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
    
        trachea_labels.append(env_thresh)
        
    return trachea_labels, trachea_mfcc_std


''' BUILD MEL DB SPECTROGRAMS '''

def get_spectrograms(signal_list,samprate,hop_length,n_fft,n_mels,fmin,fmax):

    mels = [melspectrogram(signal_list[i], sr=samprate, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax) for i in range(len(signal_list))]
    
    mel_db_list = [librosa.power_to_db(mels[i],ref=np.max) for i in range(len(mels))]

    return mel_db_list


''' SLICE SPECTROGRAMS INTO SMALL WINDOWS '''


def data_label_split(spectrogram,labels,model_window_size,model_hop_length):   
    
    # ensure window size is odd so there's a middle window frame to centre on
    assert model_window_size % 2 != 0  
    
    # move window model_hop_length frames at a time through spectrogram to create slices
    spec_sliced = [spectrogram[:,i*model_hop_length:i*model_hop_length+model_window_size] for i in range(int((spectrogram.shape[1] - model_window_size)/model_hop_length))]
    
    # label as 1 if middle frame of given spectrogram slice is 1, else 0
    labels = [1 if labels[i] == 1 else 0 for i in range(int(model_window_size/2),spectrogram.shape[1]-int(model_window_size/2)-model_hop_length,model_hop_length)]

    return spec_sliced, labels


''' LEAVE SIGNAL OUT FOR TESTING '''

def leave_signals_out(melslices_list, melslices_labels_list, test_signals):
    
    train_signals = np.delete(np.arange(0,21,1),test_signals)
    
    # create train and test sets with leave one out
    melslices_train = [x for i,x in enumerate(melslices_list) if i in train_signals]
    labels_train = [x for i,x in enumerate(melslices_labels_list) if i in train_signals]
    
    melslices_test = [x for i,x in enumerate(melslices_list) if i in test_signals]
    labels_test = [x for i,x in enumerate(melslices_labels_list) if i in test_signals]
    
    assert len(melslices_train) == len(labels_train)
    assert len(melslices_test) == len(labels_test)
    
    return melslices_train, melslices_test, labels_train, labels_test


''' STANDARDISE TRAIN AND VAL DATASET '''

def train_standardise(melslices_train,labels_train):
    
    # flatten input list of melslices for standardising
    melslices_train = [item for sublist in melslices_train for item in sublist]
    labels_train = [item for sublist in labels_train for item in sublist]
    
    # find width of each mel spectrogram window, and total length of flattened list
    window_width = melslices_train[0].shape[1]
    org_train_len = len(melslices_train)
    
    # re-combine data to standardise
    melslices_train = np.hstack(melslices_train)
    
    # standardise
    train_mean = melslices_train.mean()
    train_std = melslices_train.std()
    melslices_train_std = (melslices_train - train_mean) / train_std
    
    # split out data into slices again
    melslices_train_std = [melslices_train_std[:,i*window_width:(i+1)*window_width:] for i in range(org_train_len)]

    # convert to numpy array
    melslices_train_std = np.array(melslices_train_std)
    
    return melslices_train_std, labels_train, train_mean, train_std


''' CREATE VALIDATION SET '''

def validation_set(melslices_train_std,labels_train,split,org_length):
    
    X_train = np.empty((0,melslices_train_std[0].shape[0],melslices_train_std[0].shape[1]))
    
    for i in range(int(len(melslices_train_std)/org_length)):
        x = melslices_train_std[i*org_length:int((i+split)*org_length)]
        X_train = np.concatenate((X_train,x))
        
    y_train = [labels_train[i*org_length:int((i+split)*org_length)] for i in range(int(len(labels_train)/org_length))]
    y_train = [item for sublist in y_train for item in sublist]
    
    X_val = np.empty((0,melslices_train_std[0].shape[0],melslices_train_std[0].shape[1]))
    for i in range(int(len(melslices_train_std)/org_length)):
        x = melslices_train_std[int((i+split)*org_length):(i+1)*org_length]
        X_val = np.concatenate((X_val,x))
        
    y_val = [labels_train[int((i+split)*org_length):(i+1)*org_length] for i in range(int(len(labels_train)/org_length))]
    y_val = [item for sublist in y_val for item in sublist]

    return X_train, X_val, y_train, y_val


''' STANDARDISE TEST SET '''

def test_standardise(melslices_test,labels_test,train_mean,train_std):
    
    # flatten input list of melslices for standardising
    melslices_test = [item for sublist in melslices_test for item in sublist]
    labels_test = [item for sublist in labels_test for item in sublist]
    
    # find input mel_window size
    window_width = melslices_test[0].shape[1]
    org_test_len = len(melslices_test)
    
    # re-combine data to standardise
    melslices_test = np.hstack(melslices_test)
    
    # standardise test set with training set values
    melslices_test_std = (melslices_test - train_mean) / train_std
        
    # split out data into slices again
    melslices_test_std = [melslices_test_std[:,i*window_width:(i+1)*window_width:] for i in range(org_test_len)]

    # convert to numpy arrays
    melslices_test_std = np.array(melslices_test_std)
    
    return melslices_test_std,labels_test


#%%

''' READ IN SIGNALS, DEFINE LABELS, BUILD SPECTROGRAMS '''

filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Trachea Envelope/"
trimlen = 2600000
samprate = 44100
mfcc_thresh = 0  # graph height above which label is 1
fwd_thresh = 20  # parameter to reduce noise as signal transitions between states
hop_length=441

n_mfcc=12
fmin_mfcc=125
fmax_mfcc=2000

n_mels = 45
fmin_mel = 125
fmax_mel = 1000
n_fft = 15000

def data_gen_pipeline(filepath,trimlen,samprate,mfcc_thresh,fwd_thresh,hop_length,n_mfcc,
                      fmin_mfcc,fmax_mfcc,n_mels,fmin_mel,fmax_mel,n_fft):

    # get chest audio signals
    chest_P1 = [get_signals(filepath+"P1_Signal_{}.wav".format(i+1),trimlen) for i in range(8)]
    chest_P2 = [get_signals(filepath+"P2_Signal_{}.wav".format(i+1),trimlen) for i in range(8)]
    chest_P3 = [get_signals(filepath+"P3_Signal_{}.wav".format(i+1),trimlen) for i in range(5)]
    
    chest_all = chest_P1 + chest_P2 + chest_P3
    del chest_P1, chest_P2, chest_P3
        
    # get trachea audio signls
    trachea_P1 = [get_signals(filepath+"P1_Signal_{}_T.wav".format(i+1),trimlen) for i in range(8)]
    trachea_P2 = [get_signals(filepath+"P2_Signal_{}_T.wav".format(i+1),trimlen) for i in range(8)]
    trachea_P3 = [get_signals(filepath+"P3_Signal_{}_T.wav".format(i+1),trimlen) for i in range(5)]
    
    trachea_all = trachea_P1 + trachea_P2 + trachea_P3
    del trachea_P1, trachea_P2, trachea_P3
    
    # get labels from MFCC Band 1 Signals
    trachea_labels, trachea_mfcc_std = get_labels(trachea_all,mfcc_thresh=mfcc_thresh,samprate=samprate,fwd_thresh=fwd_thresh,n_mfcc=n_mfcc,hop_length=hop_length,fmin=fmin_mfcc,fmax=fmax_mfcc)
    
    # get mel spectrograms of trachea signals
    trachea_mel_db_list = get_spectrograms(trachea_all,samprate,hop_length,n_fft,n_mels,fmin_mel,fmax_mel)
    
    # get mel spectrograms of chest signals
    chest_mel_db_list = get_spectrograms(chest_all,samprate,hop_length,n_fft,n_mels,fmin_mel,fmax_mel)
    
    return chest_all, trachea_all, chest_mel_db_list, trachea_mel_db_list, trachea_labels, trachea_mfcc_std

chest_all,trachea_all,chest_mel_db_list, trachea_mel_db_list, trachea_labels,trachea_mfcc_std = data_gen_pipeline(filepath,trimlen,samprate,mfcc_thresh,fwd_thresh,hop_length,n_mfcc,
                      fmin_mfcc,fmax_mfcc,n_mels,fmin_mel,fmax_mel,n_fft)


#%%

''' PREP DATA INTO TRAIN, VAL, TEST SET '''

model_window_size = 15
model_hop_length = 1
test_signals = [7,15,20]
val_split = 0.8
org_length = trachea_mel_db_list[0].shape[1]

def data_prep_pipeline(model_window_size,model_hop_length,test_signals,val_split,org_length):

    melslices_list, melslices_labels_list = map(list,zip(*[data_label_split(trachea_mel_db_list[i], trachea_labels[i], model_window_size,model_hop_length) for i in range(len(trachea_mel_db_list))]))
    
    melslices_train, melslices_test, labels_train, labels_test = leave_signals_out(melslices_list, melslices_labels_list, test_signals)

    melslices_train_std, labels_train, train_mean, train_std = train_standardise(melslices_train,labels_train)
    
    X_train, X_val, y_train, y_val = validation_set(melslices_train_std,labels_train,val_split,org_length)
    
    X_test, y_test = test_standardise(melslices_test,labels_test,train_mean,train_std)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    

X_train, X_val, X_test, y_train, y_val, y_test = data_prep_pipeline(model_window_size,model_hop_length,test_signals,val_split,org_length)



#%%

''' VISUALISE LABELS AND MFCC SIGNAL '''

# vistualise to ensure labels are roughly correct
fig, axs = plt.subplots(8,3)
fig.set_figheight(30)
fig.set_figwidth(35)
plt.subplots_adjust(hspace=0.5)

for i in range(len(trachea_P1)):
    axs[i,0].plot(trachea_mfcc_normed[i],color='g')
    axs[i,0].plot(mfcc_thresholds_normed[i],color='b')
    axs[i,0].set_title('Patient 1 MFCC1 Signal {}'.format(i+1))
    axs[i,0].set_xlim(xmin=0,xmax=len(trachea_mfcc_normed[i]))
    
for i in range(len(trachea_P2)):
    axs[i,1].plot(trachea_mfcc_normed[8+i],color='g')
    axs[i,1].plot(mfcc_thresholds_normed[8+i],color='b')
    axs[i,1].set_title('Patient 2 MFCC1 Signal {}'.format(i+1))
    axs[i,1].set_xlim(xmin=0,xmax=len(trachea_mfcc_normed[i]))
    
for i in range(len(trachea_P3)):
    axs[i,2].plot(trachea_mfcc_normed[16+i],color='g')
    axs[i,2].plot(mfcc_thresholds_normed[16+i],color='b')
    axs[i,2].set_title('Patient 3 MFCC1 Signal {}'.format(i+1))
    axs[i,2].set_xlim(xmin=0,xmax=len(trachea_mfcc_normed[i]))
    

#%%
    
''' VISUALISE LABELS AND MEL SPECTROGRAMS '''

signum = 2

plt.figure(figsize=(14,4))
specshow(trachea_mel_db_list[signum],sr=samprate,x_axis='frames')
plt.plot(trachea_labels[signum]*30,color='w',linewidth=1.5)
plt.title('P1: Trachea Signal {} and Trachea MFCC Labelling'.format(signum+1),pad=12)
plt.savefig(filepath+'P1 Signal {} Trachea Mel Spectrogram'.format(signum+1),dpi=500,bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(14,4))
specshow(chest_mel_db_list[signum],sr=samprate,x_axis='frames')
plt.plot(trachea_labels[signum]*30,color='w',linewidth=1.5)
plt.title('P1: Chest Signal {} and Trachea MFCC Labelling'.format(signum+1),pad=12)
plt.savefig(filepath+'P1 Signal {} Chest Mel Spectrogram'.format(signum+1),dpi=500,bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(14,4))
plt.plot(trachea_mfcc_std[signum],color='g')
plt.plot(trachea_labels[signum],color='b')
plt.title('P1 MFCC1 Signal {}'.format(signum+1))
plt.xlim((0,len(trachea_mfcc_std[signum])))
plt.savefig(filepath+'P1 Signal {} Trachea MFCC1 & Threshold'.format(signum+1),dpi=500,bbox_inches='tight')
plt.show()
plt.close()
