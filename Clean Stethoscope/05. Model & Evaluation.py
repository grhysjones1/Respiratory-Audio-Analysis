#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:14:35 2018

@author: garethjones
"""

#%%

''' BUILD 2D CONV MODEL '''

from keras import models,layers,optimizers

model = models.Sequential()
model.add(layers.Conv2D(32,(5,3),activation='relu',input_shape=(45,15,3)))
model.add(layers.MaxPooling2D((3,1)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Flatten())
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

''' FIT MODEL ON TRAINING DATA '''

history = model.fit(
        mel_slices_train,
        labels_train,
        batch_size=32,
        epochs = 100,
        validation_data = (mel_slices_val,labels_val),
        verbose=1)


#%%

''' PLOT LOSS & ACCURACY '''

fig, axs = plt.subplots(1,2,figsize=(15,5))

axs[0].plot(history.history['loss'],label='train_loss')
axs[0].plot(history.history['val_loss'],label='val_loss')
axs[0].legend()
axs[0].set_title('Model Loss')

axs[1].plot(history.history['acc'],label='train_acc')
axs[1].plot(history.history['val_acc'],label='val_acc')
axs[1].legend()
axs[1].set_title('Model Accuracy')

plt.show()
plt.close()


#%%

''' EVALUATE ROC CURVE '''

from sklearn.metrics import roc_curve, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.utils.fixes import signature

# calculate model test predictions
y_pred = model.predict(mel_slices_rebal_test)

# calculate necessary metrics
fpr, tpr, thresholds = roc_curve(labels_rebal_test,y_pred)
precision, recall, _ = precision_recall_curve(labels_rebal_test, y_pred)

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

print('Model Precision Score = {:.1%}'.format(precision_score(labels_rebal_test, y_pred.round())))
print('Model Recall Score = {:.1%}'.format(recall_score(labels_rebal_test, y_pred.round())))
print('Model F1 Score = {:.3f}'.format(f1_score(labels_rebal_test, y_pred.round()))+' (best at 1)')
