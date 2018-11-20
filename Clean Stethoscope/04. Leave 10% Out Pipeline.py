#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:18:50 2018

@author: garethjones
"""

''' LEAVE OUT 10% ALL DATA PREPROCESSING PIPELINES '''


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
    #mel_db_labels_list = [preprocessor.label_mels_by_fft(annotations_mono[i]) for i in range(len(annotations_mono))]
    
    # label the mel spectrograms by error in time
    mel_db_labels_list = [preprocessor.label_mels_by_time(annotations_mono[i]) for i in range(len(annotations_mono))]
    
    for i in range(len(mel_db_labels_list)):
        assert len(mel_db_labels_list[i]) == mel_db_list[i].shape[1]
        
    return mel_db_list, mel_db_labels_list


def preprocess_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size):
    # split mel spectrograms into 70/20/10 for train, val, test
    datalen = len(mel_db_labels_list[0])
    
    traindata = [mel_db_list[i][:,0:int(datalen*0.7),:] for i in range(10)]
    traindata = np.array(traindata)
    trainlabels = [mel_db_labels_list[i][0:int(datalen*0.7)] for i in range(10)]
    
    valdata = [mel_db_list[i][:,int(datalen*0.7):int(datalen*0.9),:] for i in range(10)]
    valdata = np.array(valdata)
    vallabels = [mel_db_labels_list[i][int(datalen*0.7):int(datalen*0.9)] for i in range(10)]
    
    testdata = [mel_db_list[i][:,int(datalen*0.9):,:] for i in range(10)]
    testdata = np.array(testdata)
    testlabels = [mel_db_labels_list[i][int(datalen*0.9):] for i in range(10)]
    
    
    # initialise the preprocessor
    preprocessor = DataPreprocessing(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size)
    
    # split data into mel slices and associated labels
    traindata_list, trainlabels_list = map(list,zip(*[preprocessor.split_mels_labels_mid_frame(traindata[i], trainlabels[i]) for i in range(10)]))
    del traindata, trainlabels
    valdata_list, vallabels_list = map(list,zip(*[preprocessor.split_mels_labels_mid_frame(valdata[i], vallabels[i]) for i in range(10)]))
    del valdata, vallabels
    testdata_list, testlabels_list = map(list,zip(*[preprocessor.split_mels_labels_mid_frame(testdata[i], testlabels[i]) for i in range(10)]))
    del testdata, testlabels
    
    # rebalance the datasets
    traindata_rebal_list, trainlabels_rebal_list = map(list,zip(*[preprocessor.balance_data(traindata_list[i],trainlabels_list[i]) for i in range(len(traindata_list))]))
    valdata_rebal_list, vallabels_rebal_list = map(list,zip(*[preprocessor.balance_data(valdata_list[i],vallabels_list[i]) for i in range(len(traindata_list))]))
    testdata_rebal_list, testlabels_rebal_list = map(list,zip(*[preprocessor.balance_data(testdata_list[i],testlabels_list[i]) for i in range(len(traindata_list))]))
    del traindata_list, trainlabels_list
    
    # standardise the training dataset
    X_train, y_train, train_mean, train_std = preprocessor.train_standardise(traindata_rebal_list[0:10], trainlabels_rebal_list[0:10])
    del traindata_rebal_list, trainlabels_rebal_list
    
    # flatten the validation and test datasets
    valdata_rebal_list = [item for sublist in valdata_rebal_list for item in sublist]
    y_val = [item for sublist in vallabels_rebal_list for item in sublist]
    testdata_rebal_list = [item for sublist in testdata_rebal_list for item in sublist]
    y_test = [item for sublist in testlabels_rebal_list for item in sublist]
    
    # standardise the validation and test datasets
    X_val = preprocessor.test_standardise(valdata_rebal_list, train_mean, train_std)
    del valdata_rebal_list, vallabels_rebal_list
    X_test = preprocessor.test_standardise(testdata_rebal_list, train_mean, train_std)
    del testdata_rebal_list, testlabels_rebal_list

    return X_train, X_val, X_test, y_train, y_val, y_test

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
model_window_size = 25  # number of spectrogram frames to show to the model at a time


#%%

''' IMPORT AUDIO PIPELINE '''

signals_mono, annotations_mono = import_pipeline(label_error,fft_hop_length,fmin,fmax,n_mels,n_fft,fft_window_size,model_hop_length,model_window_size,filepath)


#%%

''' CREATE MEL SPECTROGRAMS PIPELINE '''

# make any changes to these variables as appropriate
label_error = 441*25  # size of error allowed in labelling respirations (in frames), set here to 250ms
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
model_window_size = 15  # number of spectrogram frames to show to the model at a time
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