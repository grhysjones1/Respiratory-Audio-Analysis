#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:31:23 2018

@author: garethjones
"""

''' FULL DATASET PREPROCESSING PIPELINES '''

# method to import audio
def import_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size,filepath):
    
    # create instance of the preprocessor
    preprocessor = DataPreprocessing(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size)
    
    # import respiratory signals, and store in a list
    signals_mono = [preprocessor.get_signals(filepath+"Signals/Signal {}.wav".format(i+1)) for i in range(10)]
    
    # import annotation signals, and store in a list
    annotations_mono = [preprocessor.get_signals(filepath+"Annotations/Annotation {}.wav".format(i+1)) for i in range(10)]
    
    # test to ensure lengths of annotation signal and respiratory signal are the same
    for i in range(len(signals_mono)):
        assert len(signals_mono[i]) == len(annotations_mono[i])
    
    # create binary annotation signals that have a single label per breath onset
    anno_gates = [preprocessor.binarize(annotations_mono[i],ampthresh['Annotation {}'.format(i+1)]) for i in range(len(annotations_mono))]
    del annotations_mono
    
    return signals_mono, anno_gates


# method to generate mel spectrograms with different variables
def melspec_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size):
    
    # create instance of the preprocessor
    preprocessor = DataPreprocessing(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size)
    
    # create stacked mel spectrograms
    mel_db_list = [preprocessor.make_stacked_mels(signals_mono[i]) for i in range(len(signals_mono))]
    
    # label the mel spectrograms by fft window size
    mel_db_labels_list = [preprocessor.label_mels_by_fft(annotations_mono[i]) for i in range(len(annotations_mono))]
    
    # label the mel spectrograms by error in time
    #mel_db_labels_list = [preprocessor.label_mels_by_time(annotations_mono[i]) for i in range(len(annotations_mono))]
    
    for i in range(len(mel_db_labels_list)):
        assert len(mel_db_labels_list[i]) == mel_db_list[i].shape[1]
        
    return mel_db_list, mel_db_labels_list


# method to preprocess mel spectrograms to send to model
def preprocess_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size):
    
    # create instance of the preprocessor
    preprocessor = DataPreprocessing(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size)
    
    # slice the mel spectrograms by model_hop_length frames fwd and model_window_size wide at a time
    # label mel slices as 1 if middle frame of spectrogram slice is also 1
    melslices_list, melslices_labels_list = map(list,zip(*[preprocessor.split_mels_labels_mid_frame(mel_db_list[i], mel_db_labels_list[i]) for i in range(10)]))
    
    # slice the mel spectrograms by model_hop_length frames fwd and model_window_size wide at a time
    # label mel slices as 1 if max of corresponding spectrogram slice is also 1
    #melslices_list, melslices_labels_list = map(list,zip(*[preprocessor.split_mels_labels_max_frame(mel_db_list[i], mel_db_labels_list[i]) for i in range(10)]))
    
    # rebalance dataset by undersampling 'negative' frames where there is no breath onset. store in a list
    melslices_rebal_list, labels_rebal_list = map(list,zip(*[preprocessor.balance_data(melslices_list[i],melslices_labels_list[i]) for i in range(len(melslices_list))]))
    del melslices_list, melslices_labels_list
    
    # test to ensure data and labels are of equal length
    for i in range(len(melslices_rebal_list)):
        assert len(melslices_rebal_list[i]) == len(labels_rebal_list[i]) # make this assertion for loop over all list elements
        
    # split data into test and train datasets
    melslices_train, melslices_test, labels_train, labels_test = preprocessor.test_split(melslices_rebal_list,labels_rebal_list)
    del melslices_rebal_list, labels_rebal_list
    
    # test to ensure data and labels are of equal length
    assert len(melslices_train) == len(labels_train)
    assert len(melslices_test) == len(labels_test)
    
    # standardise test and training datasets
    melslices_train_std, melslices_test_std = preprocessor.standardise(melslices_train, melslices_test)
    del melslices_train, melslices_test
    
    # create validation dataset from training set
    melslices_train_std, melslices_val_std, labels_train, labels_val = preprocessor.create_val_set(melslices_train_std, labels_train)
    
    # return train, val, and test sets, along with labels
    return melslices_train_std, melslices_val_std, melslices_test_std, labels_train, labels_val, labels_test


#%%
    
''' DEFINE PIPELINE VARIABLES '''

# DataPreprocessor variables to be defined
filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"  # filepath where audio signals are saved
ampthresh = {  # eyeballed thresholds for annotation  audio signals (each signal differs depending on loudness)
'Annotation 1' : 0.1, 'Annotation 2' : 0.1, 'Annotation 3' : 0.07, 'Annotation 4' : 0.1, 'Annotation 5' : 0.03,
'Annotation 6' : 0.1, 'Annotation 7' : 0.095, 'Annotation 8' : 0.1, 'Annotation 9' : 0.05, 'Annotation 10' : 0.1}
label_error = 441*25  # size of error allowed in labelling respirations (in frames), set here to 250ms
fft_hop_length = 441  # hop length (in audio frames) of fourier window when calculating mel spectrogram
fmin = 125  # min frequency limit for mel spectrograms (Hz)
fmax = 500  # max frequency limit for mel spectrograms (Hz)
n_mels = 55  # number of frequency bins high to create mel spectrogram
n_fft = [20000,21000,22000]  # fourier window sizes to create stacked spectrograms
fft_window_size = n_fft[1]  # use middle fourier window size to when windowing the annotation signal also
model_hop_length = 1  # number of frames to jump window for slicing mel spectrogram for model
model_window_size = 69  # number of spectrogram frames to show to the model at a time


#%%

''' IMPORT AUDIO PIPELINE '''

signals_mono, annotations_mono = import_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size,filepath)


#%%

''' CREATE MEL SPECTROGRAMS PIPELINE '''

# make any changes to these variables as appropriate
label_error = 441*10  # size of error allowed in labelling respirations (in frames), set here to 250ms
fft_hop_length = 441  
fmin = 125  
fmax = 500
n_mels = 55
n_fft = [20000,21000,22000]
fft_window_size = n_fft[1]

mel_db_list, mel_db_labels_list = melspec_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size)


#%%

''' PREPROCESS DATA & TRAIN WITH GRID SEARCH '''

# make any changes to these variables as appropriate
model_hop_length = [1, 4, 7]  # number of frames to jump window for slicing mel spectrogram for model
model_window_size = 69  # number of spectrogram frames to show to the model at a time
epochs = [20,60,100]

# dictionary to record grid search scores
scores = {'hop_len':[],'epochs':[],'tr_loss':[], 'tr_acc':[], 'v_loss':[], 'v_acc':[], 'acc':[], 'prec':[], 'recall':[], 'f1':[]}

for i in range(len(model_hop_length)):
    
    # create data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(
            label_error = label_error,
            fft_hop_length = fft_hop_length,
            fmin = fmin,
            fmax = fmax,
            n_mels = n_mels,
            n_fft = n_fft,
            fft_window_size = fft_window_size,
            model_hop_length = model_hop_length[i],
            model_window_size = model_window_size            
            )
    
    # build model
    from keras import models, layers, optimizers
    model = models.Sequential()
    model.add(layers.Conv2D(32,(5,3),activation='relu',input_shape=(X_train[0].shape[0],X_train[0].shape[1],X_train[0].shape[2])))
    model.add(layers.MaxPooling2D((3,1)))
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((3,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # compile model
    model.compile(
        optimizer=optimizers.RMSprop(lr=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
        )
    
    # fit model
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs[i], validation_data=(X_val,y_val), verbose=1)
    
    # predict on test data
    y_pred = model.predict(X_test)
    
    # calculate accuracy/precision metrics
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    f1 = f1_score(y_test, y_pred.round())
    
    # store metrics in dictionary
    scores['hop_len'].append(model_hop_length[i])
    scores['epochs'].append(epochs[i])
    scores['tr_loss'].append(model.history.history['loss'][-1])
    scores['tr_acc'].append(model.history.history['acc'][-1])
    scores['v_loss'].append(model.history.history['val_loss'][-1])
    scores['v_acc'].append(model.history.history['val_acc'][-1])
    scores['acc'].append(accuracy)
    scores['prec'].append(precision)
    scores['recall'].append(recall)
    scores['f1'].append(f1)


# print test scores
print()
print('For the fixed variables:')
print('fft_hop_length = {}'.format(fft_hop_length))
print('fmin = {}'.format(fmin))
print('fmax = {}'.format(fmax))
print('n_mels = {}'.format(n_mels))
print('n_fft = {}'.format(n_fft))
print('fft_window_size = {}'.format(n_fft[1]))
print()
print('Grid search scores:')
for i in range(len(model_hop_length)):
    print('hop_len = {}, epochs = {}, tr_loss = {:.3f}, tr_acc = {:.3f}, v_loss = {:.3f}, v_acc = {:.3f}, acc = {:.3f}, prec = {:.3f}, recall = {:.3f}, f1 = {:.3f}'.format(
            scores['hop_len'][i],
            scores['epochs'][i],
            scores['tr_loss'][i],
            scores['tr_acc'][i],
            scores['v_loss'][i],
            scores['v_acc'][i],
            scores['acc'][i],
            scores['prec'][i],    
            scores['recall'][i],  
            scores['f1'][i]
            ))