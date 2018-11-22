#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:34:39 2018

@author: garethjones
"""

''' CLASS FOR DATA PREPROCESSING '''

# library imports
import numpy as np
from scipy.io import wavfile
import wave
import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class DataPreprocessing:
    
    ''' METHODS FOR ALL PREPROCESSING '''
    
    # initialise class variables
    def __init__(self,label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size):
        self.samprate = 44100  # sample rate of audio signals
        self.trim_length = 2650000  # number of frames to trim audio input signals to
        self.fwdthresh = 15000  # number of frames in annotation signal to surpress to 0 after a positive label is found
        self.label_error = label_error  # size of error allowed in labelling respirations (in frames), set here to 250ms
        self.fft_hop_length = fft_hop_length  # hop length (in audio frames) 
        self.fmin = fmin  # min frequency limit for mel spectrograms
        self.fmax = fmax  # max frequency limit for mel spectrograms
        self.n_mels = n_mels  # number of frequency bins high to create mel spectrogram
        self.n_fft = n_fft  # fourier window sizes to create stacked spectrograms
        self.fft_window_size = fft_window_size  # use middle fourier window size to window the annotation signal also
        self.model_hop_length = model_hop_length
        self.model_window_size = model_window_size  # number of spectrogram frames to show to the model
    
    
    # import audio signals as wave files, and trim to a given length
    def get_signals(self,filepath):
        
        # read in audio file as numpy array
        sample = wave.open(filepath,'r')
        nframes = sample.getnframes()
        wav = wavfile.read(filepath)[1]
        signal = wav[0:nframes*2]
        
        # normalize by 16 bit width (ie 2^15)
        signal = signal / (2.**15)  
        
        # trim to given length
        signal = signal[0:self.trim_length]
        
        # transpose and make mono
        signal_mono = signal.T[0]
        
        return signal_mono
    
    
    # set annotation signals to be binary step functions, one positive label per breath onset
    def binarize(self,signal,ampthresh):
        
        # create list of 1s and 0s where annotation is above given threshold
        binarized_signal = [1 if signal[i] > ampthresh else 0 for i in range(len(signal))]
        
        # supress noise in binary signal due to anologue sound capture
        for i in range(len(binarized_signal)):
            if binarized_signal[i] == 1: 
                for j in range(1,self.fwdthresh):
                    if i+j < len(binarized_signal):
                        binarized_signal[i+j] = 0
                    else:
                        j = self.fwdthresh - i
                        binarized_signal[i+j] = 0 
    
        return binarized_signal   
    

    # create mel spectrograms that are stacked in depth dependent on number of fourier windows to calculate
    def make_stacked_mels(self,mono_signal):     
        
        # create mel spectrograms with different fft window size, all other variables the same
        mel = [melspectrogram(mono_signal, sr=self.samprate, hop_length=self.fft_hop_length, n_fft=j, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax) for j in self.n_fft]
        # turn spectrograms into log values
        mel_db = [librosa.power_to_db(mel[k],ref=np.max) for k in range(len(mel))]
        # re-stack these spectrograms into a single array
        mel_db = np.stack(mel_db,axis=-1)  
        
        return mel_db
    
    
    # reduce the length of the annotation signals to the same length as mel spectrograms, by using the fft_window size
    def label_mels_by_fft(self,annotation_signal):
        
        # get indicies of annotation array to centre search window at, given by fft_hop_length of mel spectrograms
        indices = list(np.arange(0,len(annotation_signal),self.fft_hop_length))
        mel_labels = []
        
        # for each given annotation index position, search back and forth by half the fft_window_size for any positive labels
        # if positive label is found within the window, set associated vertical slice of mel spectrogram to be positive
        for i in indices:    
            if ((i - self.fft_window_size/2) > 0) & ((i + self.fft_window_size/2) < len(annotation_signal)):
                label_window = annotation_signal[int(i-self.fft_window_size/2):int(i+self.fft_window_size/2)]
                max_label = max(label_window)
                mel_labels.append(max_label)
            
            elif (i - self.fft_window_size/2) < 0:
                label_window = annotation_signal[0:int(i+self.fft_window_size/2)]
                max_label = max(label_window)
                mel_labels.append(max_label)
            
            elif (i + self.fft_window_size/2) > len(annotation_signal):
                label_window = annotation_signal[int(i-self.fft_window_size/2):len(annotation_signal)]
                max_label = max(label_window)
                mel_labels.append(max_label)
        
        return mel_labels
    
    
    # reduce the length of the annotation signals to the same length as mel spectrograms, by using a given time error
    def label_mels_by_time(self,annotation_signal):
        
        # turn input list to np array
        annotation_signal = np.array(annotation_signal)
        
        # find where the positive labels in the annotation signal are    
        pos_label_indicies = [i for i, x in enumerate(annotation_signal) if x == 1]
        
        # set labels to 1 either side, by some time error in frames
        for i in pos_label_indicies:
            
            if ((i - self.label_error/2) > 0) & ((i + self.label_error/2) < len(annotation_signal)):
                annotation_signal[i-int(self.label_error/2):i+int(self.label_error/2)] = 1
            
            elif (i - self.label_error/2) < 0:
                annotation_signal[0:i+int(self.label_error/2)] = 1
                
            elif (i + self.label_error/2) > len(annotation_signal):
                annotation_signal[i-int(self.label_error/2):len(annotation_signal)] = 1
        
        # window annotation signal by mel_spectrogram fft_hop_length, and max each window for label
        mel_labels = [max(annotation_signal[i*self.fft_hop_length:(i+1)*self.fft_hop_length]) for i in range(int(len(annotation_signal)/self.fft_hop_length)+1)]
    
        return mel_labels
    
    
    
    # slice spectrograms into given window length, moved by model_hop_length frames through signal, to generate data to send to model
    def split_mels_labels_mid_frame(self,spectrogram,labels):   
    
        # ensure window size is odd so there's a middle window frame to centre on
        assert self.model_window_size % 2 != 0  
        
        # move window model_hop_length frames at a time through spectrogram to create slices
        spec_sliced = [spectrogram[:,i*self.model_hop_length:i*self.model_hop_length+self.model_window_size,:] for i in range(int((spectrogram.shape[1] - self.model_window_size)/self.model_hop_length))]
        
        # label as 1 if middle frame of given spectrogram slice is 1, else 0
        labels = [1 if labels[i] == 1 else 0 for i in range(int(self.model_window_size/2),spectrogram.shape[1]-int(self.model_window_size/2)-self.model_hop_length,self.model_hop_length)]
        
        return spec_sliced, labels
    
    
    # slice spectrograms into given window length, moved by model_hop_length frames through signal, to generate data to send to model
    def split_mels_labels_max_frame(self,spectrogram,labels):   
    
        # ensure window size is odd so there's a middle window frame to centre on
        assert self.model_window_size % 2 != 0  
        
        # move window model_hop_length frames at a time through spectrogram to create slices
        spec_sliced = [spectrogram[:,i*self.model_hop_length:i*self.model_hop_length+self.model_window_size,:] for i in range(int((spectrogram.shape[1] - self.model_window_size)/self.model_hop_length))]
        
        # label as 1 if corresponding window of spectrogram has a label within it
        labels = [1 if max(labels[i*self.model_hop_length:i*self.model_hop_length+self.model_window_size]) == 1 else 0 for i in range(int((spectrogram.shape[1] - self.model_window_size)/self.model_hop_length))]      
        
        return spec_sliced, labels
    
    
    # creates balanced dataset, returning a list of signals that are all positive slices followed by equal number of negative slices
    def balance_data(self,melslices,labels):
        
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
    
    
    # create validation set from training dataset
    def create_val_set(self,melslices_train_std, labels_train):
        melslices_train_std, melslices_val_std, labels_train, labels_val = train_test_split(melslices_train_std,labels_train,test_size=0.2)
        melslices_train_std = np.array(melslices_train_std)
        melslices_val_std = np.array(melslices_val_std)
        
        return melslices_train_std, melslices_val_std, labels_train, labels_val
    
    
    ''' METHODS FOR FULL DATASET PREPROCESSING '''

    # split out a test and train dataset when using full dataset
    def test_split(self,melslices,labels):
        
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
    
    
    # standardise train dataset using mean and std deviation of entire training set, use same values to standarise test set
    def standardise(self,melslices_train,melslices_test):
        
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
    
    
    
    ''' METHODS FOR LEAVE ONE OUT PREPROCESSING '''
    
    
    # standardise train dataset using mean and std deviation of entire training set, use same values to standarise test set
    def train_standardise(self,melslices_train,labels_train):
        
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
    
    
    # standardise test set using training mean and std
    def test_standardise(self,melslices_test,train_mean,train_std):
        
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
    

    
#%%