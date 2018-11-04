#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:49:09 2018

@author: garethjones
"""

#%%

''' NEED SOMEONE TO CHECK THIS CODE FOR ME '''

''' CHOP SPECTOGRAMS AND NORMALIZE '''

from sklearn.preprocessing import scale

window_len = 15

mel_slices_normed_list = []
for i in range(len(mel_db_list)):
    
    for j in range(mel_db.shape[1]-15):
        slices = mel_db_list[i][:,j:j+15,:]
        # normalize to zero mean and unit variance along rows of each spectogram within each slice
        # consider looking at this along columns as well, may improve training?
        slices_normed = [scale(slices[:,:,k],axis=1) for k in range(3)]
        # stacks elements of list back into a single numpy array
        slices_normed = np.stack(slices_normed,axis=-1)
        # puts theses single arrays into a list, length of one signal
        mel_slices_normed_list.append(slices_normed)
    
    # puts each of the lists of mel arrays into a mega list of all 10 signals
    # mel_slices_normed_list.append(mel_slices)
        

#%%

''' CREATE NEW LABELS FOR EACH FRAME '''

# looks for the labels at the middle value of each mel spectrogram slice of window size 15
labels_list_shrunk = []
for i in range(len(labels_list)):

    for j in range(int(window_len/2),mel_db.shape[1]-int(window_len/2)-1):
        
        if labels_list[i][j] == 1:
            x = 1
        else:
            x = 0
        
        labels_list_shrunk.append(x)
   
     
#%%

# see new labels vs old
labels_sums = [sum(labels_list[i]) for i in range(len(labels_list))]
labels_shrunk_sums = [sum(labels_list_shrunk[i*1025:(i+1)*1025]) for i in range(len(labels_list))]
print('Original positive labels = ' + str(labels_sums))
print('New shrunk positive labels = ' + str(labels_shrunk_sums))


#%%

''' CHOOSE RANDOM SLICES AND LABELS FOR MODELLING '''
# split data into training 70%, validation 20%, test 10%

indices = np.arange(0,len(mel_slices_normed_list))
np.random.shuffle(indices)

indices_train = indices[0:int(0.7*len(mel_slices_normed_list))]
indices_val = indices[int(0.7*len(mel_slices_normed_list)):int(0.9*len(mel_slices_normed_list))]
indices_test = indices[int(0.9*len(mel_slices_normed_list)):len(mel_slices_normed_list)]

assert len(indices_test)+len(indices_train)+len(indices_val) == len(mel_slices_normed_list)


mel_slices_train = np.array([mel_slices_normed_list[i] for i in indices_train])
labels_train = np.array([labels_list_shrunk[i] for i in indices_train])

mel_slices_val = np.array([mel_slices_normed_list[i] for i in indices_val])
labels_val = np.array([labels_list_shrunk[i] for i in indices_val])

mel_slices_test = np.array([mel_slices_normed_list[i] for i in indices_test])
labels_test = np.array([labels_list_shrunk[i] for i in indices_test])

assert len(mel_slices_train) == len(labels_train)
assert len(mel_slices_val) == len(labels_val)
assert len(mel_slices_test) == len(labels_test)


#%%

''' REBALANCE DATA '''

neg_label_indices = [index for index, x in enumerate(labels_list_shrunk) if x == 0]
pos_label_indices = [index for index, x in enumerate(labels_list_shrunk) if x == 1]
assert (len(pos_label_indices) + len(neg_label_indices)) == len(mel_slices_normed_list)

neg_mel_slices = [mel_slices_normed_list[i] for i in neg_label_indices]
pos_mel_slices = [mel_slices_normed_list[i] for i in pos_label_indices]
assert (len(neg_mel_slices) + len(pos_mel_slices)) == len(mel_slices_normed_list)

indices_to_choose = np.arange(0,len(neg_mel_slices))
indices_to_sample = np.random.choice(indices_to_choose,len(pos_mel_slices))
indices_to_sample = np.sort(indices_to_sample)


neg_mel_reduced = [neg_mel_slices[i] for i in indices_to_sample]

mel_slices_rebal = pos_mel_slices + neg_mel_reduced
labels_rebal = np.concatenate((np.ones(len(pos_mel_slices)) , np.zeros(len(neg_mel_reduced))))


indices = np.arange(0,len(mel_slices_rebal))
np.random.shuffle(indices)

indices_rebal_train = indices[0:int(0.7*len(mel_slices_rebal))]
indices_rebal_val = indices[int(0.7*len(mel_slices_rebal)):int(0.9*len(mel_slices_rebal))]
indices_rebal_test = indices[int(0.9*len(mel_slices_rebal)):len(mel_slices_rebal)]

assert len(indices_rebal_test)+len(indices_rebal_train)+len(indices_rebal_val) == len(mel_slices_rebal)


mel_slices_rebal_train = np.array([mel_slices_rebal[i] for i in indices_rebal_train])
labels_rebal_train = np.array([labels_rebal[i] for i in indices_rebal_train])

mel_slices_rebal_val = np.array([mel_slices_rebal[i] for i in indices_rebal_val])
labels_rebal_val = np.array([labels_rebal[i] for i in indices_rebal_val])

mel_slices_rebal_test = np.array([mel_slices_rebal[i] for i in indices_rebal_test])
labels_rebal_test = np.array([labels_rebal[i] for i in indices_rebal_test])

assert len(mel_slices_rebal_train) == len(labels_rebal_train)
assert len(mel_slices_rebal_val) == len(labels_rebal_val)
assert len(mel_slices_rebal_test) == len(labels_rebal_test)


#%%

print(sum(labels_rebal_train)/len(labels_rebal_train))
print(sum(labels_rebal_val)/len(labels_rebal_val))
print(sum(labels_rebal_test)/len(labels_rebal_test))