#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:38:58 2018

@author: garethjones
"""

#%%

''' IMPORTS & GLOBALS '''

from PIL import Image
import numpy as np
filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/Spectograms/"
test_specto = Image.open(filepath+'Signal 1.png','r')
test_specto = np.array(test_specto)

width_specto = test_specto.shape[1]  # pixel width of original spectograms
npixels = 24  # how many pixels per slice to have (24 equates to 250 slices)
n_pixel_splits = int(width_specto/npixels)  # equates to 250 slices
n_frames_split = int(len(anno_gates[0])/n_pixel_splits) # number of frames in annotation signals to split at
print(n_frames_split)


#%%

''' SPLIT ANNOTATION GATES INTO LABELS '''

# cut the labels list into equal segments and take the max
labels = []
for i in range(len(anno_gates)):
    for j in range(int(len(anno_gates[i])/n_frames_split)):
        x = max(anno_gates[i][j*n_frames_split:(j+1)*n_frames_split])
        labels.append(x)


#%%

''' IMPORT SPECTOGRAMS AND CHOP '''

for i in range(len(signals)):
    img = Image.open(filepath+'Signal {}.png'.format(i+1),'r')
    img = np.array(img)
    
    for j in range(n_pixel_splits):
        imgslice = img[:,j*npixels:(j+1)*npixels,0:3] # dumps final dimension of 255s
        imgslice = Image.fromarray(imgslice)
        
        if labels[(i*250)+j] == 1:
            imgslice.save(filepath+'Slices Tight Labels/positive_slice{}.png'.format(i*250+j),'PNG')
        else:
            imgslice.save(filepath+'Slices Tight Labels/negative_slice{}.png'.format(i*250+j),'PNG')

            
#%%
            
''' CREATE TRAIN AND TEST FOLDERS '''

import os, shutil

original_dataset_dir = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/Spectograms/Slices Tight Labels"

dirs = []

base_dir = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/Spectograms/"
dirs.append(base_dir)

train_dir = os.path.join(base_dir, "train_tight_labels")
dirs.append(train_dir)
val_dir = os.path.join(base_dir, "validation_tight_labels")
dirs.append(val_dir)
test_dir = os.path.join(base_dir, "test_tight_labels")
dirs.append(test_dir)

train_positive_dir = os.path.join(train_dir, "positive")
dirs.append(train_positive_dir)
train_negative_dir = os.path.join(train_dir, "negative")
dirs.append(train_negative_dir)

val_positive_dir = os.path.join(val_dir, "positive")
dirs.append(val_positive_dir)
val_negative_dir = os.path.join(val_dir, "negative")
dirs.append(val_negative_dir)

test_positive_dir = os.path.join(test_dir, "positive")
dirs.append(test_positive_dir)
test_negative_dir = os.path.join(test_dir, "negative")
dirs.append(test_negative_dir)

for directory in dirs:
    if not os.path.exists(directory):
       os.mkdir(directory)


#%%

 ''' RANDOM SHUFFLE IMG NAMES '''

positivenames = ['positive_slice{}.png'.format(i) for i in range(len(labels)) if labels[i] == 1]
negativenames = ['negative_slice{}.png'.format(i) for i in range(len(labels)) if labels[i] == 0]
print('Number positive images = '+str(len(positivenames)))
print('Number negative images = '+str(len(negativenames)))

# randomly shuffle the lists
np.random.shuffle(positivenames)
np.random.shuffle(negativenames)


#%%

''' SPLIT INTO TEST AND TRAIN DATASETS '''

# Randomly choose 70% for training, 15% for validation, 15% for testing
trainsplit = 0.7
valsplit = 0.15
testsplit = 0.15

# select 70% positive training images
for i in range(int(len(positivenames)*trainsplit)):
    imgname = positivenames[i]
    src = os.path.join(original_dataset_dir, imgname)
    dest = os.path.join(train_positive_dir, imgname)
    shutil.copy(src, dest)
    
# select 15% positive validation images
for i in range(int(len(positivenames)*trainsplit),int(len(positivenames)*(trainsplit+valsplit))):
    imgname = positivenames[i]
    src = os.path.join(original_dataset_dir, imgname)
    dest = os.path.join(val_positive_dir, imgname)
    shutil.copy(src, dest)
    
# select 15% positive test images
for i in range(int(len(positivenames)*(trainsplit+valsplit)),len(positivenames)):
    imgname = positivenames[i]
    src = os.path.join(original_dataset_dir, imgname)
    dest = os.path.join(test_positive_dir, imgname)
    shutil.copy(src, dest)

# select 70% negative training images
for i in range(int(len(negativenames)*trainsplit)):
    imgname = negativenames[i]
    src = os.path.join(original_dataset_dir, imgname)
    dest = os.path.join(train_negative_dir, imgname)
    shutil.copy(src, dest)
    
# select 15% negative validation images
for i in range(int(len(negativenames)*trainsplit),int(len(negativenames)*(trainsplit+valsplit))):
    imgname = negativenames[i]
    src = os.path.join(original_dataset_dir, imgname)
    dest = os.path.join(val_negative_dir, imgname)
    shutil.copy(src, dest)

# select 15% negative test images
for i in range(int(len(negativenames)*(trainsplit+valsplit)),len(negativenames)):
    imgname = negativenames[i]
    src = os.path.join(original_dataset_dir, imgname)
    dest = os.path.join(test_negative_dir, imgname)
    shutil.copy(src, dest)

for directory in dirs:
    print(directory, ":", len(os.listdir(directory)))

print("Done.")


#%%

print('Train positives = '+str(len(os.listdir(train_positive_dir))))
print('Train negatives = '+str(len(os.listdir(train_negative_dir))))
print('Val positives = '+str(len(os.listdir(val_positive_dir))))
print('Val negatives = '+str(len(os.listdir(val_negative_dir))))
print('Test positives = '+str(len(os.listdir(test_positive_dir))))
print('Test negatives = '+str(len(os.listdir(test_negative_dir))))
print('Proportion positive = '+str(len(os.listdir(train_positive_dir))/(len(os.listdir(train_negative_dir))+len(os.listdir(train_positive_dir)))))
