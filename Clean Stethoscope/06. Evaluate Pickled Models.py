#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:25:45 2018

@author: garethjones
"""

#%%
''' READ IN PICKLED MODEL '''

# google drive filepath
gfilepath = '/Users/garethjones/Google Drive/01. Data Science/01. Respiratory Analysis/03. Pickled Models/'

import pickle
model001 = pickle.load(open(gfilepath+'model001_fscore976.pkl','rb'))
history001 = pickle.load(open(gfilepath+'history001_fscore976.pkl','rb'))
melparams001 = pickle.load(open(gfilepath+'melparams001_fscore976.pkl','rb'))



#%%

''' PLOT TRAINING & VALIDATION LOSS & ACCURACY '''

fig, axs = plt.subplots(1,2,figsize=(15,5))

axs[0].plot(history001['loss'],label='train_loss')
axs[0].plot(history001['val_loss'],label='val_loss')
axs[0].legend()
axs[0].set_title('Model Loss')

axs[1].plot(history001['acc'],label='train_acc')
axs[1].plot(history001['val_acc'],label='val_acc')
axs[1].legend()
axs[1].set_title('Model Accuracy')

plt.show()
plt.close()



#%%

''' PREDICT USING MODEL '''

# choose balance or unbalanced data
X_test = mel_slices_rebal_test
y_test = labels_rebal_test

# calculate model test predictions
y_pred = model001.predict(X_test)



#%%

''' EVALUATE MODEL '''

from sklearn.metrics import roc_curve, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.fixes import signature

# calculate necessary metrics
print('Model Accuracy = {:.3f}'.format(accuracy_score(y_test, y_pred.round()))) 
print('Model Precision = {:.3f}'.format(precision_score(y_test, y_pred.round()))) # how many of the frames it said were positive actually were
print('Model Recall = {:.3f}'.format(recall_score(y_test, y_pred.round())))  # how many of the breath onset frames in total the model managed to identify
print('Model F1 = {:.3f}'.format(f1_score(y_test, y_pred.round())))



''' PRINT ROC CURVE '''

# calculate necessary metrics
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# plot ROC and precision/recall curve
fig,axs = plt.subplots(1,2,figsize=(15,5))

axs[0].plot([0, 1], [0, 1], 'k--')
axs[0].plot(fpr, tpr)
axs[0].set_xlabel('False positive rate')
axs[0].set_ylabel('True positive rate')
axs[0].set_title('ROC Curve')

step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
axs[1].step(recall, precision, color='b', alpha=0.2, where='post')
axs[1].fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('Precision/Recall curve')

plt.show()
plt.close()
