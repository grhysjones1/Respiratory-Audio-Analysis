#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:58:53 2018

@author: garethjones
"""

''' METHODS FOR PRE-PROCESSING DATA '''

# import audio signals as wave files, and trim to a given length
def get_signals(filepath,trim_length):
    
    # read in audio file as numpy array
    sample = wave.open(filepath,'r')
    nframes = sample.getnframes()
    wav = wavfile.read(filepath)[1]
    signal = wav[0:nframes*2]
    
    # normalize by 16 bit width (ie 2^15)
    signal = signal / (2.**15)  
    
    # trim to given length
    signal = signal[0:trim_length]
    
    # transpose and make mono
    signal_mono = signal.T[0]
    
    return signal_mono


# set annotation signals to be binary step functions, one positive signal per breath onset
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


# create mel spectrograms that are stacked dependent on number of fourier windows to calculate
def make_stacked_mels(mono_signal,n_fft,samprate,hop_length,fmin,fmax,n_mels):     
    
    # create mel spectrograms with different fft window size, all other variables the same
    mel = [melspectrogram(mono_signal, sr=samprate, hop_length=hop_length, n_fft=j, n_mels=n_mels, fmin=fmin, fmax=fmax) for j in n_fft]
    # turn spectrograms into log values
    mel_db = [librosa.power_to_db(mel[k],ref=np.max) for k in range(len(mel))]
    # re-stack these spectrograms into a single array
    mel_db = np.stack(mel_db,axis=-1)  
    
    return mel_db


# reduce the length of the annotation signals to the same length as mel spectrograms
def reduce_annotations(annotation_signal,hop_length,fft_window_size):
    
    # get indicies of annotation array to centre search window at, given by hop_length of mel spectrograms
    indices = list(np.arange(0,len(annotation_signal),hop_length))
    labels = []
    
    # for each given annotation index position, search back and forth by half the fft_window_size for any positive labels
    # if positive label is found within the window, set associated time slice of mel spectrogram to be positive
    for i in indices:    
        if ((i - fft_window_size/2) > 0) & ((i + fft_window_size/2) < len(annotation_signal)):
            label_window = annotation_signal[int(i-fft_window_size/2):int(i+fft_window_size/2)]
            max_label = max(label_window)
            labels.append(max_label)
        
        elif (i - fft_window_size/2) < 0:
            label_window = annotation_signal[0:int(i+fft_window_size/2)]
            max_label = max(label_window)
            labels.append(max_label)
        
        elif (i + fft_window_size/2) > len(annotation_signal):
            label_window = annotation_signal[int(i-fft_window_size/2):len(annotation_signal)]
            max_label = max(label_window)
            labels.append(max_label)
    
    return labels


# slice spectrograms into given window length, moved one frame at a time through signal, to generate data to send to model
def data_label_split(spectrogram,labels,model_window_size):   

    # ensure window size is odd so there's a middle window frame to centre on
    assert model_window_size % 2 != 0  
    
    # move window 1 frame at a time through spectrogram to create slices
    spec_sliced = [spectrogram[:,i:i+model_window_size,:] for i in range(spectrogram.shape[1] - model_window_size)]
    
    # select label from labels list where index is the position of middle frame in spectrogram slice
    labels = [1 if labels[i] == 1 else 0 for i in range(int(model_window_size/2),spectrogram.shape[1]-int(model_window_size/2)-1)]
    
    return spec_sliced, labels


# creates balanced dataset, returning a list of signals that are all positive slices followed by equal number of negative slices
def balance_data(melslices,labels):
    
    # find indices of positive & negative labels
    neg_label_indices = [i for i, x in enumerate(labels) if x == 0]
    pos_label_indices = [i for i, x in enumerate(labels) if x == 1]
    
    # find mel spectrogram slices with associated positive & negative labels
    neg_melslices = [melslices[i] for i in neg_label_indices]
    pos_melslices = [melslices[i] for i in pos_label_indices]
    
    # sample fewer negative (majority) slices to be same number as positive slices
    neg_indices = np.arange(0,len(neg_melslices))
    neg_indices_sample = np.random.choice(neg_indices,len(pos_melslices),replace=False)
    neg_indices_sample = np.sort(neg_indices_sample)
    
    # create down-sampled majority class dataset
    neg_mel_reduced = [neg_melslices[i] for i in neg_indices_sample]
    
    # concatenate positive and negative data together
    melslices_rebal = pos_melslices + neg_mel_reduced
    
    # create new labels for rebalanced dataset
    labels_rebal = np.concatenate((np.ones(len(pos_melslices)) , np.zeros(len(neg_mel_reduced))))
    
    return melslices_rebal, labels_rebal


# split out a test and train dataset
def test_split(melslices,labels):
    
    # flatten lists of melslices for splitting
    melslices = [item for sublist in melslices for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    
    # split data by 10%
    from sklearn.model_selection import train_test_split
    melslices_train, melslices_test, labels_train, labels_test = train_test_split(melslices,labels,test_size=0.1)
    
    # turn to numpy arrays
    melslices_train = np.array(melslices_train)
    melslices_test = np.array(melslices_test)
    
    return melslices_train, melslices_test, labels_train, labels_test


# normalise train dataset using mean value of entire training set, use same values to normalise test set
def mean_normalise(melslices_train, melslices_test):
    
    # find input mel_window size
    window_width = melslices_train[0].shape[1]
    org_train_len = len(melslices_train)
    org_test_len = len(melslices_test)
    
    # re-combine data to normalize
    melslices_train = np.hstack(melslices_train)
    melslices_test = np.hstack(melslices_test)
    
    # normalize 
    melslices_train_min = melslices_train.min(axis=(0,1),keepdims=True)
    melslices_train_max = melslices_train.max(axis=(0,1),keepdims=True)
    melslices_train_mean = melslices_train.mean(axis=(0,1),keepdims=True)
    melslices_train_normed = (melslices_train - melslices_train_mean) / (melslices_train_max - melslices_train_min)
    
    # normalize test data with train_val values
    melslices_test_normed = (melslices_test - melslices_train_mean) / (melslices_train_max - melslices_train_min)
    
    # split out data into slices again
    melslices_train_normed = [melslices_train_normed[:,i*window_width:(i+1)*window_width:,:] for i in range(org_train_len)]
    melslices_test_normed = [melslices_test_normed[:,i*window_width:(i+1)*window_width:,:] for i in range(org_test_len)]
    
    return melslices_train_normed, melslices_test_normed


# standardise train dataset using mean and std deviation of entire training set, use same values to standarise test set
def standardise(melslices_train,melslices_test):
    
    # find input mel_window size
    window_width = melslices_train[0].shape[1]
    org_train_len = len(melslices_train)
    org_test_len = len(melslices_test)
    
    # re-combine data to normalize
    melslices_train = np.hstack(melslices_train)
    melslices_test = np.hstack(melslices_test)
    
    # standardise
    train_mean = melslices_train.mean(axis=(0,1),keepdims=True)
    train_std = melslices_train.std(axis=(0,1),keepdims=True)
    melslices_train_std = (melslices_train - train_mean) / train_std

    # standardise test set with training set values
    melslices_test_std = (melslices_test - train_mean) / train_std
    
    # split out data into slices again
    melslices_train_std = [melslices_train_std[:,i*window_width:(i+1)*window_width:,:] for i in range(org_train_len)]
    melslices_test_std = [melslices_test_std[:,i*window_width:(i+1)*window_width:,:] for i in range(org_test_len)]

    # convert to numpy arrays
    melslices_train_std = np.array(melslices_train_std)
    melslices_test_std = np.array(melslices_test_std)
    
    return melslices_train_std, melslices_test_std


def create_val_set(melslices_train_std, labels_train):
    from sklearn.model_selection import train_test_split
    melslices_train_std, melslices_val_std, labels_train, labels_val = train_test_split(melslices_train_std,labels_train,test_size=0.22)
    melslices_train_std = np.array(melslices_train_std)
    melslices_val_std = np.array(melslices_val_std)
    
    return melslices_train_std, melslices_val_std, labels_train, labels_val


# combine all functions into one moster method
def master_preprocessing():
    
    # import respiratory signals
    signals_mono = [get_signals(filepath+"Signals/Signal {}.wav".format(i+1),trim_length) for i in range(10)]
    
    # import annotation signals
    annotations_mono = [get_signals(filepath+"Annotations/Annotation {}.wav".format(i+1),trim_length) for i in range(10)]
    
    # test to ensure lengths are the same
    for i in range(len(signals_mono)):
        assert len(signals_mono[i]) == len(annotations_mono[i])
    
    # create binary annotations
    anno_gates = [binarize(annotations_mono[i],thresholds['Annotation {}'.format(i+1)],fwdthresh) for i in range(len(annotations_mono))]
    del annotations_mono
    
    # create mel spectrograms
    mel_db_list = [make_stacked_mels(signals_mono[i], n_fft, samprate, hop_length, fmin, fmax, n_mels) for i in range(len(signals_mono))]
    del signals_mono
    
    # create labels for mel spectrograms from annotation signals
    mel_db_labels_list = [reduce_annotations(anno_gates[i],hop_length,fft_window_size) for i in range(len(anno_gates))]
    del anno_gates
    
    # test to ensure labels are same lengths as mel spectrograms
    for i in range(len(mel_db_labels_list)):
        assert len(mel_db_labels_list[i]) == mel_db_list[i].shape[1]

    # window the mel spectrograms and labels, move by one frame fwd at a time, to send to model for training
    melslices_list, melslices_labels_list = map(list,zip(*[data_label_split(mel_db_list[i], mel_db_labels_list[i], model_window_size) for i in range(len(mel_db_list))]))
    del mel_db_list, mel_db_labels_list
    
    # rebalance the dataset by undersampling 'negative' frames where there is no breath onset
    melslices_rebal, melslices_labels_rebal = map(list,zip(*[balance_data(melslices_list[i],melslices_labels_list[i]) for i in range(len(melslices_list))]))
    del melslices_list, melslices_labels_list
    
    # split data into test and train datasets
    melslices_train, melslices_test, labels_train, labels_test = test_split(melslices_rebal,melslices_labels_rebal)
    del melslices_rebal, melslices_labels_rebal
    
    # test to ensure data and labels are of equal length
    assert len(melslices_train) == len(labels_train)
    assert len(melslices_test) == len(labels_test)
    
    # standardise test and training datasets
    melslices_train_std, melslices_test_std = standardise(melslices_train, melslices_test)
    
    return melslices_train_std, melslices_test_std, labels_train, labels_test



#%%

''' IMPORTS AND DEFINE GLOBAL VARIABLES '''

# library imports
from scipy.io import wavfile
import wave
import numpy as np
import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
import warnings
warnings.filterwarnings('ignore')

# vairables to be defined
samprate = 44100  # sample rate of audio signals
trim_length = 2650000  # number of frames to trim audio input signals to
fwdthresh = 15000  # used to reduce noise in number of frames after a annotation label is found
hop_length = 441  # hop length (in audio frames) 
fmin = 125  # min frequency limit for mel spectrograms
fmax = 500  # max frequency limit for mel spectrograms
n_mels = 55  # number of frequency bins high to create mel spectrogram
n_fft = [20000,21000,22000]  # fourier window sizes to create stacked spectrograms
fft_window_size = n_fft[1]  # use middle fourier window size to window the annotation signal also
model_window_size = 69  # number of spectrogram frames to show to the model
filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"  # filepath where audio signals are saved
thresholds = {  # eyeballed thresholds for annotation signals
        'Annotation 1' : 0.1, 'Annotation 2' : 0.1, 'Annotation 3' : 0.07, 'Annotation 4' : 0.1, 'Annotation 5' : 0.03,
        'Annotation 6' : 0.1, 'Annotation 7' : 0.095, 'Annotation 8' : 0.1, 'Annotation 9' : 0.05, 'Annotation 10' : 0.1}


''' RUN MASTER METHOD '''

X_train, X_test, y_train, y_test = master_preprocessing()



#%%

''' MEL SPECTROGRAM GRID SEARCH '''

hop_length = [221,441,882]
n_mels = [30,45,60]
fmax = [500,700,900]

for i in range(len(hop_length)):
    












