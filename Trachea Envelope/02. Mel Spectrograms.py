#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 00:17:22 2018

@author: garethjones
"""


''' IMPORTS & GLOBALS '''

import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
from sklearn.model_selection import train_test_split
import pylab
import warnings
warnings.filterwarnings('ignore')

samprate = 44100
hop_length = 441
fmin = 125
fmax = 1000
n_mels = 45
n_fft = 4410

''' SPECTROGRAMS '''

mels = [melspectrogram(trachea_mono[i], sr=samprate, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax) for i in range(len(trachea_mono))]

mel_db_list = [librosa.power_to_db(mels[i],ref=np.max) for i in range(len(mels))]


#%%

''' VISUALISE MEL SPECTROGRAMS FOR ONE SIGNAL '''

signal_num = 1

plt.figure(figsize=(10,4))
specshow(mel_db[signal_num],sr=samprate,x_axis='frames')
plt.plot(mfcc_thresholds_normed[signal_num]*20)
plt.show()
plt.close()

#%%

# slice spectrograms and labels
def data_label_split(spectrogram,labels,model_window_size):   

    # ensure window size is odd so there's a middle window frame to centre on
    assert model_window_size % 2 != 0  
    
    # move window 1 frame at a time through spectrogram to create slices
    spec_sliced = [spectrogram[:,i:i+model_window_size] for i in range(spectrogram.shape[1] - model_window_size)]
    
    # select label from labels list where index is the position of middle frame in spectrogram slice
    labels = [1 if labels[i] == 1 else 0 for i in range(int(model_window_size/2),spectrogram.shape[1]-int(model_window_size/2)-1)]
    
    return spec_sliced, labels

melslices_list, melslices_labels_list = map(list,zip(*[data_label_split(mel_db_list[i], mfcc_thresholds_normed[i], 15) for i in range(len(mel_db_list))]))


#%%

# create train and test sets with leave one out
melslices_train = melslices_list[:-1]
labels_train = melslices_labels_list[:-1]

melslices_test = melslices_list[-1]
labels_test = melslices_labels_list[-1]

#%%

# standardise data
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

melslices_train_std, labels_train, train_mean, train_std = train_standardise(melslices_train,labels_train)


#%%
from sklearn.model_selection import train_test_split
# create validation set from training dataset
def create_val_set(melslices_train_std, labels_train):
    melslices_train_std, melslices_val_std, labels_train, labels_val = train_test_split(melslices_train_std,labels_train,test_size=0.2)
    melslices_train_std = np.array(melslices_train_std)
    melslices_val_std = np.array(melslices_val_std)
    
    return melslices_train_std, melslices_val_std, labels_train, labels_val


#%%
# create validation dataset from training set
melslices_train_std, melslices_val_std, labels_train, labels_val = create_val_set(melslices_train_std, labels_train)

#%%
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
    melslices_test_std = [melslices_test_std[:,i*window_width:(i+1)*window_width:] for i in range(org_test_len)]

    # convert to numpy arrays
    melslices_test_std = np.array(melslices_test_std)
    
    return melslices_test_std


# select 10th audio signal as test, and standardise using training mean and std
melslices_test_std = test_standardise(melslices_test, train_mean, train_std)

