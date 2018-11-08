#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:49:09 2018

@author: garethjones
"""

#%%

''' CHOP SPECTOGRAMS AND NORMALIZE '''

from sklearn.preprocessing import scale

window_len = 69  # this should always be odd so there's a middle window frame

mel_slices_normed_list = []
for i in range(len(mel_db_list)):
    
    for j in range(mel_db.shape[1] - window_len):
        slices = mel_db_list[i][:,j:j+window_len,:]
        
        # normalize to zero mean and unit variance along rows of each spectogram within each slice
        # consider looking at this along rows as well, may improve training? or normalize whole mel slice?
        slices_normed = [scale(slices[:,:,k],axis=1) for k in range(3)]
        
        # stacks elements of list back into a single numpy array
        slices_normed = np.stack(slices_normed,axis=-1)
        
        # puts theses single arrays into a list, length of one signal
        mel_slices_normed_list.append(slices_normed)


''' CREATE NEW LABELS FOR EACH FRAME '''

# looks for the labels at the middle value of each mel spectrogram slice of given window size
labels_list_shrunk = []
for i in range(len(labels_list)):

    for j in range(int(window_len/2),mel_db.shape[1]-int(window_len/2)-1):
        
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

''' FOR IMBALANCED DATA: CHOOSE RANDOM SLICES AND LABELS '''

from sklearn.model_selection import train_test_split

# split data by 10%, then 22.2% to get roughly 70:20:10 split
mel_slices_train, mel_slices_test, labels_train, labels_test = train_test_split(mel_slices_normed_list,labels_list_shrunk,test_size=0.1)
mel_slices_train, mel_slices_val, labels_train, labels_val = train_test_split(mel_slices_train,labels_train,test_size=0.222)

# turn to numpy arrays
mel_slices_train = np.array(mel_slices_train)
mel_slices_val = np.array(mel_slices_val)
mel_slices_test = np.array(mel_slices_test)

assert len(mel_slices_train) == len(labels_train)
assert len(mel_slices_val) == len(labels_val)
assert len(mel_slices_test) == len(labels_test)


''' TOTAL SIZE & POS/NEG SPLIT '''

print('Samples in Training Data = '+str(len(mel_slices_train)))
print('Samples in Validation Data = '+str(len(mel_slices_val)))
print('Samples in Test Data = '+str(len(mel_slices_test)))
print('Pos/Neg Ratio in Training = '+str('{:.3f}'.format(sum(labels_train)/len(labels_train))))
print('Pos/Neg Ratio in Training = '+str('{:.3f}'.format(sum(labels_val)/len(labels_val))))
print('Pos/Neg Ratio in Training = '+str('{:.3f}'.format(sum(labels_test)/len(labels_test))))


#%%

''' FOR BALANCED DATA: CHOOSE RANDOM SLICES AND LABELS '''

# find indices of positive & negative labels
neg_label_indices = [i for i, x in enumerate(labels_list_shrunk) if x == 0]
pos_label_indices = [i for i, x in enumerate(labels_list_shrunk) if x == 1]
assert (len(pos_label_indices) + len(neg_label_indices)) == len(mel_slices_normed_list)

# find mel spectrogram slices with associated positive & negative labels
neg_mel_slices = [mel_slices_normed_list[i] for i in neg_label_indices]
pos_mel_slices = [mel_slices_normed_list[i] for i in pos_label_indices]
assert (len(neg_mel_slices) + len(pos_mel_slices)) == len(mel_slices_normed_list)

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


''' TOTAL SIZE & POS/NEG SPLIT '''

print('Samples in Training Data = '+str(len(mel_slices_rebal_train)))
print('Samples in Validation Data = '+str(len(mel_slices_rebal_val)))
print('Samples in Test Data = '+str(len(mel_slices_rebal_test)))
print('Pos/Neg Ratio in Training = '+str('{:.3f}'.format(sum(labels_rebal_train)/len(labels_rebal_train))))
print('Pos/Neg Ratio in Training = '+str('{:.3f}'.format(sum(labels_rebal_val)/len(labels_rebal_val))))
print('Pos/Neg Ratio in Training = '+str('{:.3f}'.format(sum(labels_rebal_test)/len(labels_rebal_test))))
