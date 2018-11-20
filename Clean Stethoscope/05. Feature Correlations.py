#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:30:15 2018

@author: garethjones
"""

''' CORRELATE MFCC AND LABELS'''

import matplotlib.pyplot as plt
from scipy import stats
filepath = "/users/garethjones/Documents/Data Science/Feebris/Data/Clean Stethoscope/"

# get middle mel of each stack
mels = [mel_db_list[i][:,:,1].T for i in range(10)]

# calculate spearman correlation matrix
spearmans = [stats.spearmanr(mels[i],mel_db_labels_list[i])[0][-1] for i in range(10)]
del mels

# select only those correlations associated with the label
spearmans = [np.delete(spearmans[i],-1) for i in range(10)]

# visualise
plt.figure()
for i in range(10):
    plt.plot(spearmans[i],label='Signal {}'.format(i+1))
plt.ylim(-0.5,0.5)
plt.xlim(0,len(spearmans[0]))
plt.title('Spearman Correlation of Mel Frequency Bands to Label\n\n 4410 Frames Label Error',pad=15)
plt.xlabel('Mel Frequency Bands (125-500Hz)',labelpad=10,size=10)
plt.ylabel('Spearman Correlation to Label',labelpad=5,size=10)
plt.legend(bbox_to_anchor=(1.02, 1.03))
plt.savefig(filepath+'Spearman Correlation - 4410 Frame Label Error.png',bbox_inches='tight',dpi=600)
plt.show()
plt.close()


#%%