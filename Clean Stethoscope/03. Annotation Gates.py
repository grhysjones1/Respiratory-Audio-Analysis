#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:08:39 2018

@author: garethjones
"""

#%%

''' GLOBAL VARIABLES & REFERENCE '''

# known clicks per sample
originalclickspersignal = {
        'Annotation 1' : 24, 'Annotation 2' : 20, 'Annotation 3' : 24, 'Annotation 4' : 34, 'Annotation 5' : 30,
        'Annotation 6' : 31, 'Annotation 7' : 32, 'Annotation 8' : 36, 'Annotation 9' : 33, 'Annotation 10' : 29
        }

# these are number of clicks per sample after chopping data
newclickspersignal = {
        'Annotation 1' : 22, 'Annotation 2' : 19, 'Annotation 3' : 22, 'Annotation 4' : 33, 'Annotation 5' : 28,
        'Annotation 6' : 30, 'Annotation 7' : 31, 'Annotation 8' : 29, 'Annotation 9' : 29, 'Annotation 10' : 26
        }

# eye-balled thresholds for each signal
thresholds = {
        'Annotation 1' : 0.1, 'Annotation 2' : 0.1, 'Annotation 3' : 0.07, 'Annotation 4' : 0.1, 'Annotation 5' : 0.03,
        'Annotation 6' : 0.1, 'Annotation 7' : 0.095, 'Annotation 8' : 0.1, 'Annotation 9' : 0.05, 'Annotation 10' : 0.1
        }


#%% 

''' MAKE ANNOTATION SIGNALS MONO '''

# make annotations one channel, transposed, absolute
annotations_mono = []
for i in range(len(annotations)):
    x = abs(annotations[i].T[0])
    annotations_mono.append(x)


#%%

''' TURN SIGNALS INTO STEP FUNCTIONS '''

# denote whenever amplitude is above threshold
anno_gates = []
for i in range(len(annotations_mono)):
    gate_list = []
    for j in range(len(annotations_mono[i])):
        if annotations_mono[i][j] > thresholds['Annotation {}'.format(i+1)]: # this is amplitude threshold
            x = 1
        else:
            x = 0
        gate_list.append(x)
    anno_gates.append(gate_list)


#%%

''' SUPRESS NOISE IN STEP FUNCTION '''

# ensure noise is removed so there's exact number of clicks
fwd_frame_thresh = 15000
size_anno_gates = []
for i in range(len(anno_gates)):
    for j in range(len(anno_gates[i])):
        if anno_gates[i][j] == 1:
            for k in range(1,fwd_frame_thresh): # this is the forward threshold for silencing frames
                if j+k < len(anno_gates[i]):
                    anno_gates[i][j+k] = 0
                else:
                    k = fwd_frame_thresh - j
                    anno_gates[i][j+k] = 0       
    size_anno_gates.append(sum(anno_gates[i]))

print(np.r_[list(newclickspersignal.values())] - np.r_[size_anno_gates])


#%%

''' VISUALISE '''

# visualise gates to ensure they're correct
fig, axs = plt.subplots(10, 1, figsize=(8,25))
for i in range(len(anno_gates)):
    axs[i].plot(anno_gates[i])
    axs[i].plot(annotations_mono[i])
    axs[i].set_title('Annotation Gate {}'.format(i+1))
plt.tight_layout()
plt.show()
plt.close()


#%%

''' EXPAND WIDTH OF STEPS '''

expand_window = int(samprate/6) # equates to roughly a third second error

# get list of indices where anno_gate signal is 1
index_list = []
for i in range(len(anno_gates)):
    indices = []
    for j in range(len(anno_gates[i])):
        if anno_gates[i][j] == 1:
            indices.append(j)
    index_list.append(indices)

# for my list of indices, expand the 'width' of each annotation gate signal
for i in range(len(index_list)):
    for j in index_list[i]:
        if j + expand_window < len(anno_gates[i]):
            for k in range(j-expand_window,j+expand_window): # this is the range within which I expand the 1s
                anno_gates[i][k] = 1
        else:
            for k in range(j-expand_window,len(anno_gates[i])):
                anno_gates[i][k] = 1


#%%
            
''' VISUALISE '''

# visualise new anno_gates to ensure they're correct
fig, axs = plt.subplots(10, 1, figsize=(8,25))
for i in range(len(anno_gates)):
    axs[i].plot(anno_gates[i])
    axs[i].plot(annotations_mono[i])
    axs[i].set_title('Annotation Gate {}'.format(i+1))
plt.tight_layout()
plt.show()
plt.close()