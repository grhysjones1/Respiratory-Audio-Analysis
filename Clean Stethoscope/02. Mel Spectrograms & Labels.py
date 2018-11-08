#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:28:49 2018

@author: garethjones
"""

#%%

''' IMPORTS & GLOBALS '''

import librosa
from librosa.display import specshow
from librosa.feature import melspectrogram
import pylab
import warnings
warnings.filterwarnings('ignore')



''' DEFINE FUNCTIONS '''

def make_stacked_mels(mono_signal,n_fft,samprate,hop_length,fmin,fmax,n_mels):     
    # create 3 mel spectrograms with different fft window size, all other variables the same
    mel = [melspectrogram(signals_mono[i], sr=samprate, hop_length=hop_length, n_fft=j, n_mels=n_mels, fmin=fmin, fmax=fmax) for j in n_fft]
    # turn spectrograms into log values
    mel_db = [librosa.power_to_db(mel[k],ref=np.max) for k in range(len(mel))]
    # re-stack these spectrograms into a single array
    mel_db = np.stack(mel_db,axis=-1)  
    return mel_db


def reduce_annotations(annotation_signal,hop_length,window_size):
    
    indices = list(np.arange(0,len(annotation_signal),hop_length))
    
    labels = []
    
    for i in indices:    
        if ((i - window_size/2) > 0) & ((i + window_size/2) < len(annotation_signal)):
            label_window = annotation_signal[int(i-window_size/2):int(i+window_size/2)]
            max_label = max(label_window)
            labels.append(max_label)
        
        elif (i - window_size/2) < 0:
            label_window = annotation_signal[0:int(i+window_size/2)]
            max_label = max(label_window)
            labels.append(max_label)
        
        elif (i + window_size/2) > len(annotation_signal):
            label_window = annotation_signal[int(i-window_size/2):len(annotation_signal)]
            max_label = max(label_window)
            labels.append(max_label)
    
    return labels



''' DEFINE VALUES AND RUN FUNCTIONS '''

samprate = 44100
hop_length = 441
fmin = 125
fmax = 500
n_mels = 55
n_fft = [20000,21000,22000]
window_size = n_fft[1]

mel_db_list = [make_stacked_mels(signals_mono[i],n_fft) for i in range(len(signals_mono))]
labels_list = [reduce_annotations(anno_gates[i],hop_length,window_size) for i in range(len(anno_gates))]

for i in range(len(labels_list)):
    assert len(labels_list[i]) == mel_db_list[i].shape[1]

print('Shape of mel spectrograms = '+str(mel_db_list[0].shape))



#%%

''' VISUALISE MEL SPECTROGRAMS FOR ONE SIGNAL '''

signal_num = 0

plt.figure(figsize=(10,8))
plt.subplots_adjust(hspace=0.63)
plt.suptitle('Mel Spectrogram for Signal #{}\nhop_length={}, n_mels={}, fmin={}, fmax={}'.format(signal_num,hop_length,n_mels,fmin,fmax),weight='bold')

for i in range(len(n_fft)):
    plt.subplot(3,1,i+1)
    plt.title(' FFT Window Length = {}'.format(n_fft[i]),weight='bold',pad=10)
    specshow(mel_db_list[signal_num][:,:,i],sr=samprate,x_axis='frames')
#plt.savefig(filepath+'FFTWindow {} hop_length {}.png'.format(str(n_fft),hop_length),dpi=900)
plt.show()
plt.close()


#%%
''' VISUALISE BINARY SIGNAL AND MEL SIZED ANNOTATIONS '''

signal_num = 1

fig,axs = plt.subplots(2,1,figsize=(12,8))
plt.subplots_adjust(hspace=0.4)
plt.suptitle('Updated Labels for Mel Spectrograms Signal #{}'.format(signal_num),weight='bold')

axs[0].plot(anno_gates[signal_num])
axs[0].set_title('Original Binary Annotation Signal',pad=10)
axs[0].set_xlim(xmin=0,xmax=len(anno_gates[signal_num]))
axs[0].set_ylim(ymin=0)
axs[0].spines['top'].set_color('none')
axs[0].spines['right'].set_color('none')

axs[1].plot(labels_list[signal_num])
axs[1].set_title('New Annotations at Mel Spectrogram Frame Size',pad=10)
axs[1].set_xlim(xmin=0,xmax=len(labels_list[signal_num]))
axs[1].set_ylim(ymin=0)
axs[1].spines['top'].set_color('none')
axs[1].spines['right'].set_color('none')

plt.show()
plt.close()



#%%

''' SAVE SPECTROGRAMS TO FILE TO SEE DIFFERENCES '''

'''
# 1s = 44,100    0.5s = 22,050   0.25s = 11,025   0.125s = 5513    0.0625s = 2756

#hop_frames = [1250,2550,5500,11000,22050]
hop_frames = [256,512,1250,2550,5500]
fft_width = [8000, 11025, 14000]
fmin = 125
fmax = 500
freq_bins = [30, 35, 40, 45]
signal = signals_mono[2]

for i in fft_width:
    for j in freq_bins:    
        plt.figure(figsize=(15,15))
        for k in range(1,6):
            meltest = melspectrogram(signal,samprate,hop_length=hop_frames[k-1],n_fft=i,fmin=fmin,fmax=fmax,n_mels=j)
            meltest_db = librosa.power_to_db(meltest,ref=np.max)
            plt.subplot(5,1,k)
            plt.title('Hop Frames = {}'.format(hop_frames[k-1]))
            specshow(meltest_db,sr=samprate)
        plt.suptitle('FFT Window Width = {} / Number Frequency Bin = {}'.format(i,j))
        plt.savefig(filepath+'Comparison FFTWindow {} Freqbins {}.png'.format(i,j),dpi=900)
        plt.show()
        plt.close()
'''