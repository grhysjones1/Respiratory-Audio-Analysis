#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:35:55 2018

@author: garethjones
"""

filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Trachea Envelope/"
trimlen = 2600000
samprate = 44100
mfcc_thresh = 0  # graph height above which label is 1
fwd_thresh = 20  # parameter to reduce noise as signal transitions between states
hop_length=441

n_mfcc=12
fmin_mfcc=125
fmax_mfcc=2000

n_mels = 45
fmin_mel = 125
fmax_mel = 1000
n_fft = 15000

# get chest audio signals
chest_P1 = [get_signals(filepath+"P1_Signal_{}.wav".format(i+1),trimlen) for i in range(8)]
chest_P2 = [get_signals(filepath+"P2_Signal_{}.wav".format(i+1),trimlen) for i in range(8)]
chest_P3 = [get_signals(filepath+"P3_Signal_{}.wav".format(i+1),trimlen) for i in range(5)]

chest_all = chest_P1 + chest_P2 + chest_P3
del chest_P1, chest_P2, chest_P3
    
# get trachea audio signls
trachea_P1 = [get_signals(filepath+"P1_Signal_{}_T.wav".format(i+1),trimlen) for i in range(8)]
trachea_P2 = [get_signals(filepath+"P2_Signal_{}_T.wav".format(i+1),trimlen) for i in range(8)]
trachea_P3 = [get_signals(filepath+"P3_Signal_{}_T.wav".format(i+1),trimlen) for i in range(5)]

trachea_all = trachea_P1 + trachea_P2 + trachea_P3
del trachea_P1, trachea_P2, trachea_P3

# get labels from MFCC Band 1 Signals
trachea_labels, trachea_mfcc_std = get_labels(trachea_all,mfcc_thresh=mfcc_thresh,samprate=samprate,fwd_thresh=fwd_thresh,n_mfcc=n_mfcc,hop_length=hop_length,fmin=fmin_mfcc,fmax=fmax_mfcc)


#%%

samprate = 44100
n_mfcc = 64
fmin = 125
fmax = 2000
hop_length = 441

chest_mfcc = [librosa.feature.mfcc(chest_all[i], sr=samprate, n_mfcc=n_mfcc,hop_length=hop_length,fmin=fmin,fmax=fmax) for i in range(len(chest_all))]

filtnorm_mfcc_list = []
for i in range(len(chest_mfcc)):
    filtnorm_mfcc = []
    for j in range(n_mfcc):
        mfcc_band = chest_mfcc[i][j,:]
        filt_mfcc_band = savgol_filter(mfcc_band, 201, 2)
        filtnorm_mfcc_band = (filt_mfcc_band-filt_mfcc_band.mean())/filt_mfcc_band.std()
        filtnorm_mfcc.append(filtnorm_mfcc_band)
    filtnorm_mfcc = np.array(filtnorm_mfcc)
    filtnorm_mfcc_list.append(filtnorm_mfcc)


#%%

signum = 1
mfcc_band = 5
graph_len = 5000

plt.figure(figsize=(10,4))
#plt.plot(chest_mfcc[signum][mfcc_band][0:graph_len],color='b')
plt.plot(filtnorm_mfcc_list[signum][mfcc_band][0:graph_len])
plt.plot(trachea_labels[signum][0:graph_len])
plt.xlim((0,graph_len))


#%%

''' PREP DATA FOR RANDOM FOREST '''

test_signals = [7,15,20]
train_signals = np.delete(np.arange(0,21,1),test_signals)

mfccs_train = [x for i,x in enumerate(filtnorm_mfcc_list) if i in train_signals]
labels_train = [x for i,x in enumerate(trachea_labels) if i in train_signals]

mfccs_test = [x for i,x in enumerate(filtnorm_mfcc_list) if i in test_signals]
labels_test = [x for i,x in enumerate(trachea_labels) if i in test_signals]

#%%

X_train = np.empty(0)


#%%

''' IMPLEMENT RANDOM FOREST '''

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, random_state=123)
rfc.fit(X_train,y_train)

acc = rfc.score(X_test,y_test)
print("Model accuracy is {:.1%}".format(acc))
