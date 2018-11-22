#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:13:47 2018

@author: garethjones
"""


''' DEFINE FUNCTIONS '''

def data_label_split(spectrogram,labels,model_window_size):   
    assert window_size % 2 != 0  # this should always be odd so there's a middle window frame
    
    # move window 1 frame at a time through spectrogram to create slices
    spec_sliced = [spectrogram[:,i:i+model_window_size,:] for i in range(spectrogram.shape[1] - model_window_size)]
    
    # select label from labels list where index is the position of middle frame in spectrogram slice
    labels = [1 if labels[i] == 1 else 0 for i in range(int(model_window_size/2),spectrogram.shape[1]-int(model_window_size/2)-1)]
    
    return spec_sliced, labels



# returns balanced mel-slices, all positive slices followed by equal number of negative slices
def balance_data(melslices,labels):
    
    # find indices of positive & negative labels
    neg_label_indices = [i for i, x in enumerate(labels) if x == 0]
    pos_label_indices = [i for i, x in enumerate(labels) if x == 1]
    #assert (len(pos_label_indices) + len(neg_label_indices)) == len(mel_slices_list)
    
    # find mel spectrogram slices with associated positive & negative labels
    neg_melslices = [melslices[i] for i in neg_label_indices]
    pos_melslices = [melslices[i] for i in pos_label_indices]
    #assert (len(neg_melslices) + len(pos_melslices)) == len(mel_slices_list)
    
    # sample fewer negative (majority) classes
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



def test_split(melslices,labels):
    
    # flatten lists of melslices for splitting
    melslices = [item for sublist in melslices for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    
    # split data by 10%, then 22.2% to get roughly 70:20:10 split
    from sklearn.model_selection import train_test_split
    melslices_train, melslices_test, labels_train, labels_test = train_test_split(melslices,labels,test_size=0.1)
    
    # turn to numpy arrays
    mel_slices_train = np.array(melslices_train)
    mel_slices_test = np.array(melslices_test)
    
    return melslices_train, melslices_test, labels_train, labels_test



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
    
    return melslices_train_std, melslices_test_std



#%%

''' DEFINE VARIABLES AND RUN '''
window_size = 69  

# map and zip(*) allow list comprehension on function with two outputs, first stored as tuples, and then mapped to lists
melslices_list, melslices_labels_list = map(list,zip(*[data_label_split(mel_db_list[i], mel_db_labels_list[i], window_size) for i in range(len(mel_db_list))]))

melslices_rebal, melslices_labels_rebal = map(list,zip(*[balance_data(melslices_list[i],melslices_labels_list[i]) for i in range(len(melslices_list))]))

melslices_train, melslices_test, labels_train, labels_test = test_split(melslices_rebal,melslices_labels_rebal)

assert len(melslices_train) == len(labels_train)
assert len(melslices_test) == len(labels_test)

#melslices_train_normed, melslices_test_normed = mean_normalise(melslices_train, melslices_test)
melslices_train_std, melslices_test_std = standardise(melslices_train, melslices_test)


#%%


''' CALCULATE DIFFERENCE IN NUMBER OF LABELS '''

# number of indices a single signal lasts for
signal_segment = int(len(labels_list_shrunk)/len(labels_list))

labels_sums = [sum(labels_list[i]) for i in range(len(labels_list))]
labels_shrunk_sums = [sum(labels_list_shrunk[i*signal_segment:(i+1)*signal_segment]) for i in range(len(labels_list))]
print('Original positive labels = ' + str(labels_sums))
print('New shrunk positive labels = ' + str(labels_shrunk_sums))

