#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:42:03 2018

@author: garethjones
"""

''' LEAVE ONE OUT APPROACH '''

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


# set annotation signals to be binary step functions, one positive label per breath onset
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


# create mel spectrograms that are stacked in depth dependent on number of fourier windows to calculate
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
    # if positive label is found within the window, set associated vertical slice of mel spectrogram to be positive
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


# slice spectrograms into given window length, moved by model_hop_length frames through signal, to generate data to send to model
def data_label_split(spectrogram,labels,model_hop_length,model_window_size):   

    # ensure window size is odd so there's a middle window frame to centre on
    assert model_window_size % 2 != 0  
    
    # move window 1 frame at a time through spectrogram to create slices
    spec_sliced = [spectrogram[:,i*model_hop_length:i*model_hop_length+model_window_size,:] for i in range(int((spectrogram.shape[1] - model_window_size)/model_hop_length))]
    
    # select label from labels list where index is the position of middle frame in spectrogram slice
    labels = [1 if labels[i] == 1 else 0 for i in range(int(model_window_size/2),spectrogram.shape[1]-int(model_window_size/2)-model_hop_length,model_hop_length)]
    
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
    neg_indices_sample = np.random.choice(neg_indices,len(pos_melslices))
    neg_indices_sample = np.sort(neg_indices_sample)
    
    # create down-sampled majority class dataset
    neg_mel_reduced = [neg_melslices[i] for i in neg_indices_sample]
    
    # concatenate positive and negative data together
    melslices_rebal = pos_melslices + neg_mel_reduced
    
    # create new labels for rebalanced dataset
    labels_rebal = np.concatenate((np.ones(len(pos_melslices)) , np.zeros(len(neg_mel_reduced))))
    
    return melslices_rebal, labels_rebal


# standardise train dataset using mean and std deviation of entire training set, use same values to standarise test set
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
    train_mean = melslices_train.mean(axis=(0,1),keepdims=True)
    train_std = melslices_train.std(axis=(0,1),keepdims=True)
    melslices_train_std = (melslices_train - train_mean) / train_std
    
    # split out data into slices again
    melslices_train_std = [melslices_train_std[:,i*window_width:(i+1)*window_width:,:] for i in range(org_train_len)]

    # convert to numpy array
    melslices_train_std = np.array(melslices_train_std)
    
    return melslices_train_std, labels_train, train_mean, train_std


# create validation set from training dataset
def create_val_set(melslices_train_std, labels_train):
    melslices_train_std, melslices_val_std, labels_train, labels_val = train_test_split(melslices_train_std,labels_train,test_size=0.2)
    melslices_train_std = np.array(melslices_train_std)
    melslices_val_std = np.array(melslices_val_std)
    
    return melslices_train_std, melslices_val_std, labels_train, labels_val


# standardise test set using training mean and std
def test_standardise(melslices_test,train_mean,train_std):
    
    # find input mel_window size
    window_width = melslices_test[0].shape[1]
    org_test_len = len(melslices_test)
    
    # re-combine data to standardise
    melslices_test = np.hstack(melslices_test)
    
    # standardise test set with training set values
    melslices_test_std = (melslices_test - train_mean) / train_std
        
    # split out data into slices again
    melslices_test_std = [melslices_test_std[:,i*window_width:(i+1)*window_width:,:] for i in range(org_test_len)]

    # convert to numpy arrays
    melslices_test_std = np.array(melslices_test_std)
    
    return melslices_test_std



# combine all functions into one moster method for preprocessing audio signals to be model ready
def master_preprocessing():
    
    ''' METHODS FOR PREPROCESSING DATA '''
    
    # import respiratory signals, and store in a list
    signals_mono = [get_signals(filepath+"Signals/Signal {}.wav".format(i+1),trim_length) for i in range(10)]
    
    # import annotation signals, and store in a list
    annotations_mono = [get_signals(filepath+"Annotations/Annotation {}.wav".format(i+1),trim_length) for i in range(10)]
    
    # test to ensure lengths of annotation signal and respiratory signal are the same
    for i in range(len(signals_mono)):
        assert len(signals_mono[i]) == len(annotations_mono[i])
    
    # create binary annotation signals that have a single label per breath onset
    anno_gates = [binarize(annotations_mono[i],ampthresh['Annotation {}'.format(i+1)],fwdthresh) for i in range(len(annotations_mono))]
    del annotations_mono
    
    # create mel spectrograms for all respiratory signals in list
    mel_db_list = [make_stacked_mels(signals_mono[i], n_fft, samprate, hop_length, fmin, fmax, n_mels) for i in range(len(signals_mono))]
    del signals_mono
    
    # create labels for each vertical slice of the mel spectrograms
    # use same size of window and hop length as used to calculate the spectrograms
    mel_db_labels_list = [reduce_annotations(anno_gates[i],hop_length,fft_window_size) for i in range(len(anno_gates))]
    del anno_gates
    
    # test to ensure new labels are same lengths as mel spectrograms
    for i in range(len(mel_db_labels_list)):
        assert len(mel_db_labels_list[i]) == mel_db_list[i].shape[1]
        
    # window the mel spectrograms and labels, move by one frame fwd at a time, to send to model for training
    melslices_list, melslices_labels_list = map(list,zip(*[data_label_split(mel_db_list[i], mel_db_labels_list[i], model_hop_length, model_window_size) for i in range(10)]))
    del mel_db_list, mel_db_labels_list
    
    # select 9/10 signals for training and rebalance dataset by undersampling 'negative' frames where there is no breath onset. store in a list
    melslices_rebal_list, labels_rebal_list = map(list,zip(*[balance_data(melslices_list[i],melslices_labels_list[i]) for i in range(len(melslices_list))]))
    del melslices_list, melslices_labels_list
    
    # test to ensure data and labels are of equal length
    for i in range(len(melslices_rebal_list)):
        assert len(melslices_rebal_list[i]) == len(labels_rebal_list[i]) # make this assertion for loop over all list elements
        
    
    ''' THE FOLLOWING METHODS NOW LEAVE ONE FULL AUDIO SIGNAL OUT FOR TESTING '''
    
    # select 9/10 audio signals and standardise for training, return numpy array of all mel slices to train on with labels
    melslices_train_std, labels_train, train_mean, train_std = train_standardise(melslices_rebal_list[0:9], labels_rebal_list[0:9])
    
    # select 10th audio signal as test, and standardise using training mean and std
    melslices_test_std = test_standardise(melslices_rebal_list[9], train_mean, train_std)
    
    # select 10th annotation signal as test labels
    labels_test = labels_rebal_list[9]
    del melslices_rebal_list, labels_rebal_list
    
    # create validation dataset from training set
    melslices_train_std, melslices_val_std, labels_train, labels_val = create_val_set(melslices_train_std, labels_train)

    # return train, val, and test sets, along with labels
    return melslices_train_std, melslices_val_std, melslices_test_std, labels_train, labels_val, labels_test


    
#%%

''' IMPORTS AND DEFINE GLOBAL VARIABLES '''

# library imports
from scipy.io import wavfile
import wave
import numpy as np
import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# vairables to be defined
samprate = 44100  # sample rate of audio signals
trim_length = 2650000  # number of frames to trim audio input signals to
fwdthresh = 15000  # number of frames in annotation signal to surpress to 0 after a positive label is found
hop_length = 441  # hop length (in audio frames) of fourier window when calculating mel spectrogram
fmin = 125  # min frequency limit for mel spectrograms (Hz)
fmax = 500  # max frequency limit for mel spectrograms (Hz)
n_mels = 55  # number of frequency bins high to create mel spectrogram
n_fft = [20000,21000,22000]  # fourier window sizes to create stacked spectrograms
fft_window_size = n_fft[1]  # use middle fourier window size to when windowing the annotation signal also
model_hop_length = 7  # number of frames to jump window for slicing mel spectrogram for model
model_window_size = 15  # number of spectrogram frames to show to the model at a time
filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"  # filepath where audio signals are saved
ampthresh = {  # eyeballed thresholds for annotation  audio signals (each signal differs depending on loudness)
        'Annotation 1' : 0.1, 'Annotation 2' : 0.1, 'Annotation 3' : 0.07, 'Annotation 4' : 0.1, 'Annotation 5' : 0.03,
        'Annotation 6' : 0.1, 'Annotation 7' : 0.095, 'Annotation 8' : 0.1, 'Annotation 9' : 0.05, 'Annotation 10' : 0.1}


''' RUN MASTER METHOD '''

X_train, X_val, X_test, y_train, y_val, y_test = master_preprocessing()

