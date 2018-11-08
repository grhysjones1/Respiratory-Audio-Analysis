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


window_size = 69  

# map and zip(*) allow me to do a list comprehension on function with two outputs
# they are first stored as tuples, and then mapped to lists
melslices_list, melslices_labels_list = map(list,zip(*[data_label_split(mel_db_list[i], mel_db_labels_list[i], window_size) for i in range(len(mel_db_list))]))


#%%


# returns balanced mel-slices, all positive slices followed by equal number of negative slices
def balanced_data(melslices,labels):
    
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
    neg_indices_sample = np.random.choice(neg_indices,len(pos_melslices))
    neg_indices_sample = np.sort(neg_indices_sample)
    
    # create down-sampled majority class dataset
    neg_mel_reduced = [neg_melslices[i] for i in neg_indices_sample]
    
    # concatenate positive and negative data together
    melslices_rebal = pos_melslices + neg_mel_reduced
    
    # create new labels for rebalanced dataset
    labels_rebal = np.concatenate((np.ones(len(pos_melslices)) , np.zeros(len(neg_mel_reduced))))
    
    return melslices_rebal, labels_rebal


#melslices_list = np.hstack(melslices_list)
#melslices_labels_list = np.hstack(melslices_labels_list)

melslices_rebal, melslices_labels_rebal = map(list,zip(*[balanced_data(melslices_list[i],melslices_labels_list[i]) for i in range(len(melslices_list))]))


#%%

def train_test_val_split(melslices,labels)
    
    # split data by 10%, then 22.2% to get roughly 70:20:10 split
    from sklearn.model_selection import train_test_split
    melslices_train, melslices_test, labels_train, labels_test = train_test_split(mel_slices_rebal,labels_rebal,test_size=0.1)
    melslices_train, melslices_val, labels_train, labels_val = train_test_split(mel_slices_train,labels_train,test_size=0.222)
    
    # turn to numpy arrays
    mel_slices_rebal_train = np.array(mel_slices_train)
    mel_slices_rebal_val = np.array(mel_slices_val)
    mel_slices_rebal_test = np.array(mel_slices_test)
    
    return melslices_train, melslices_val, melslices_test, labels_train, labels_val, labels_test
    

melslices_train, melslices_val, melslices_test, labels_train, labels_val, labels_test = train_test_val_split(melslices_rebal,melslices_labels_rebal)


#%%

''' RUN ASSERTIONS HERE '''

assert len(mel_slices_rebal_train) == len(labels_rebal_train)
assert len(mel_slices_rebal_val) == len(labels_rebal_val)
assert len(mel_slices_rebal_test) == len(labels_rebal_test)



#%%

''' CHOP SPECTROGRAMS '''

mel_slices_list = []
for i in range(len(mel_db_list)):
    for j in range(mel_db_list[0].shape[1] - window_len):
        slices = mel_db_list[i][:,j:j+window_len,:]
        mel_slices_list.append(slices)
        

#%%
        
''' CALCULATE LABELS '''

# looks for the labels at the middle value of each mel spectrogram slice of given window size
labels_list_shrunk = []
for i in range(len(labels_list)):

    for j in range(int(window_len/2),mel_db_list[0].shape[1]-int(window_len/2)-1):
        
        if labels_list[i][j] == 1:
            x = 1
        else:
            x = 0
        
        labels_list_shrunk.append(x)


''' CALCULATE DIFFERENCE IN NUMBER OF LABELS '''

# number of indices a single signal lasts for
signal_segment = int(len(labels_list_shrunk)/len(labels_list))

labels_sums = [sum(labels_list[i]) for i in range(len(labels_list))]
labels_shrunk_sums = [sum(labels_list_shrunk[i*signal_segment:(i+1)*signal_segment]) for i in range(len(labels_list))]
print('Original positive labels = ' + str(labels_sums))
print('New shrunk positive labels = ' + str(labels_shrunk_sums))


#%%

''' FOR BALANCED DATA: CHOOSE RANDOM SLICES AND LABELS '''

# find indices of positive & negative labels
neg_label_indices = [i for i, x in enumerate(labels_list_shrunk) if x == 0]
pos_label_indices = [i for i, x in enumerate(labels_list_shrunk) if x == 1]
assert (len(pos_label_indices) + len(neg_label_indices)) == len(mel_slices_list)

# find mel spectrogram slices with associated positive & negative labels
neg_mel_slices = [mel_slices_list[i] for i in neg_label_indices]
pos_mel_slices = [mel_slices_list[i] for i in pos_label_indices]
assert (len(neg_mel_slices) + len(pos_mel_slices)) == len(mel_slices_list)

# sample fewer negative (majority) classes
neg_indices = np.arange(0,len(neg_mel_slices))
neg_indices_sample = np.random.choice(neg_indices,len(pos_mel_slices))
neg_indices_sample = np.sort(neg_indices_sample)

# create down-sampled majority class dataset
neg_mel_reduced = [neg_mel_slices[i] for i in neg_indices_sample]

# concatenate positive and negative data together
mel_slices_rebal = pos_mel_slices + neg_mel_reduced

# create new labels for rebalanced dataset
labels_rebal = np.concatenate((np.ones(len(pos_mel_slices)) , np.zeros(len(neg_mel_reduced))))

# split data by 10%, then 22.2% to get roughly 70:20:10 split
from sklearn.model_selection import train_test_split
mel_slices_rebal_train, mel_slices_rebal_test, labels_rebal_train, labels_rebal_test = train_test_split(mel_slices_rebal,labels_rebal,test_size=0.1)
mel_slices_rebal_train, mel_slices_rebal_val, labels_rebal_train, labels_rebal_val = train_test_split(mel_slices_rebal_train,labels_rebal_train,test_size=0.222)

# turn to numpy arrays
mel_slices_rebal_train = np.array(mel_slices_rebal_train)
mel_slices_rebal_val = np.array(mel_slices_rebal_val)
mel_slices_rebal_test = np.array(mel_slices_rebal_test)

assert len(mel_slices_rebal_train) == len(labels_rebal_train)
assert len(mel_slices_rebal_val) == len(labels_rebal_val)
assert len(mel_slices_rebal_test) == len(labels_rebal_test)


del labels_rebal, mel_slices_rebal, neg_mel_reduced, neg_mel_slices, pos_label_indices, pos_mel_slices, slices


#%%

''' NORMALISE TRAINING & VALIDATION '''

# re-combine data to normalize
train_combined = np.hstack(mel_slices_rebal_train)
val_combined = np.hstack(mel_slices_rebal_val)
test_combined = np.hstack(mel_slices_rebal_test)

# combine train and val data to normalize together
train_val_combined = np.hstack((train_combined,val_combined))

# normalize 
train_val_min = train_val_combined.min(axis=(0,1),keepdims=True)
train_val_max = train_val_combined.max(axis=(0,1),keepdims=True)
train_val_normed = (train_val_combined - train_val_min) / (train_val_max - train_val_min)

# normalize test data with train_val values
test_normed = (test_combined - train_val_min) / (train_val_max - train_val_min)

# split out train and val again
train_normed = train_val_normed[:,0:train_combined.shape[1],:]
val_normed = train_val_normed[:,train_combined.shape[1]:,:]


del train_combined, val_combined, test_combined, train_val_combined, train_val_normed


#%%

''' CHOP SPECTROGRAMS AGAIN '''

X_train = np.array(np.split(train_normed,mel_slices_rebal_train.shape[0],axis=1))
X_val = np.array(np.split(val_normed,mel_slices_rebal_val.shape[0],axis=1))
X_test = np.array(np.split(test_normed,mel_slices_rebal_test.shape[0],axis=1))

y_train = labels_rebal_train
y_val = labels_rebal_val
y_test = labels_rebal_test

del train_normed, val_normed, test_normed