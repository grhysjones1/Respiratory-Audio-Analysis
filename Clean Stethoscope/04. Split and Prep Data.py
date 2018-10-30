#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:38:58 2018

@author: garethjones
"""

#%%

''' SPLIT ANNOTATION GATES INTO LABELS '''

# work out exact segments to split on
nframes = len(anno_gates[0])
width = img.shape[1]
nsplits = 24
windowsplit = nframes/(width/nsplits)
print(windowsplit)
windowsplit = int(windowsplit)

# cut the labels list into equal segments and take the max
labels = []
for i in range(len(anno_gates)):
    for j in range(int(len(anno_gates[i])/windowsplit)):
        x = max(anno_gates[i][j*windowsplit:(j+1)*windowsplit])
        labels.append(x)


#%%

''' IMPORT SPECTOGRAMS AND CHOP '''
from PIL import Image
import numpy as np

filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"
npixels = int(width/nsplits)

for i in range(len(labels_list)):
    img = Image.open(filepath+'Spectograms/Signal {}.png'.format(i+1),'r')
    img = np.array(img)
    for j in range(npixels):
        imgslice = img[:,j*nsplits:(j+1)*nsplits,:]
        imgslice = Image.fromarray(imgslice)
        imgslice.save(filepath+'Spectograms/Slices/slice{}_{}.png'.format(i+1,j+1),'PNG')
