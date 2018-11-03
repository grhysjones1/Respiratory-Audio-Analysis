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


mel_slices_train = [mel_slices_normed_list[i] for i in indices_train]
labels_train = np.asarray([labels_list_shrunk[i] for i in indices_train])

mel_slices_val = [mel_slices_normed_list[i] for i in indices_val]
labels_val = np.asarray([labels_list_shrunk[i] for i in indices_val])

mel_slices_test = [mel_slices_normed_list[i] for i in indices_test]
labels_test = np.asarray([labels_list_shrunk[i] for i in indices_test])

assert len(mel_slices_train) == len(labels_train)
assert len(mel_slices_val) == len(labels_val)
assert len(mel_slices_test) == len(labels_test)


#%%

from keras import models,layers,optimizers

model = models.Sequential()
model.add(layers.Conv2D(32,(3,7),activation='relu',input_shape=(45,15,3)))
model.add(layers.MaxPooling2D((3,1)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((3,1)))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )


#%%

def generator(
    data,
    lookback, 
    delay,
    min_index, 
    max_index,
    shuffle=False, 
    batch_size=batch_size,
    step=6):

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback # TODO redundant?
   
    while True:

        if shuffle:
            rows = np.random.randint(
                min_index + lookback,
                max_index,
                batch_size
            )
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets

      
#%%
        
def generator(data,labels,batch_size=32):
    while True:
        for i in range(0,len(data),batch_size):
            slices = data[i:i+batch_size]
            labels_temp = np.array(labels[i:i+batch_size])
            yield (slices, labels_temp)
        
train_gen = generator(mel_slices_train,labels_train,32)


#%%

history = model.fit_generator(
        train_gen,
        steps_per_epoch = 225,
        epochs = 100
        )

