#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:32:06 2018

@author: garethjones
"""

#%%

''' CREATE IMAGE GENERATOR '''

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size = (3525,24),
    batch_size = 25,
    class_mode = "binary")

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size = (3525,24),
    batch_size = 25,
    class_mode = "binary")

test_generator = datagen.flow_from_directory(
        test_dir,
        target_size = (3525,24),
        batch_size = 25,
        class_mode = 'binary')


#%%

''' VISUALISE GENERATOR '''
images = next(train_generator)
print(images[0].shape)
print(len(images))

images = next(val_generator)
print(images[0].shape)
print(len(images))

images = next(test_generator)
print(images[0].shape)
print(len(images))


#%%

''' BUILD A CONV NEURAL NET '''

from keras import models, layers, optimizers

model_1 = models.Sequential()
model_1.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(3525,24,3)))
model_1.add(layers.MaxPooling2D((2,2)))
model_1.add(layers.Conv2D(64, (3,3), activation="relu"))
model_1.add(layers.MaxPooling2D((2,2)))
model_1.add(layers.Conv2D(128, (3,3), activation="relu"))
model_1.add(layers.MaxPooling2D((2,2)))
model_1.add(layers.Flatten())
model_1.add(layers.Dense(128, activation="relu"))
model_1.add(layers.Dropout(0.25))
model_1.add(layers.Dense(1, activation="sigmoid"))
model_1.summary()

model_1.compile(
    optimizer=optimizers.RMSprop(lr=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

#%%

''' FIT MODEL '''

history_1 = model_1.fit_generator(
    train_generator,
    steps_per_epoch = 70, # 1749 training pics, 25 batch size, need 70 steps per epoch
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 15, # same as the comment above
    verbose = 1
)


#%%

''' VISUALISE ACCURACY '''

fig, axs = plt.subplots(1,2,figsize=(15,5))

axs[0].plot(history_1.history['loss'],label='train_loss')
axs[0].plot(history_1.history['val_loss'],label='val_loss')
axs[0].legend()
axs[0].set_title('Model Loss')

axs[1].plot(history_1.history['acc'],label='train_acc')
axs[1].plot(history_1.history['val_acc'],label='val_acc')
axs[1].legend()
axs[1].set_title('Model Accuracy')

plt.show()
plt.close()


#%%

''' TRY SLIGHTLY DIFFERENT MODEL '''

model_2 = models.Sequential()
model_2.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(3525,24,3)))
model_2.add(layers.MaxPooling2D((2,2)))
model_2.add(layers.Conv2D(32, (3,3), activation="relu"))
model_2.add(layers.MaxPooling2D((2,2)))
model_2.add(layers.Conv2D(64, (3,3), activation="relu"))
model_2.add(layers.MaxPooling2D((2,2)))
model_2.add(layers.Flatten())
model_2.add(layers.Dense(512, activation="relu"))
model_2.add(layers.Dense(1, activation="sigmoid"))
model_2.summary()

model_2.compile(
    optimizer=optimizers.RMSprop(lr=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

history_2 = model_2.fit_generator(
    train_generator,
    steps_per_epoch = 70, # 1749 training pics, 25 batch size, need 70 steps per epoch
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 15, # same as the comment above
    verbose = 1
)

''' VISUALISE '''

fig, axs = plt.subplots(1,2,figsize=(15,5))

axs[0].plot(history_2.history['loss'],label='train_loss')
axs[0].plot(history_2.history['val_loss'],label='val_loss')
axs[0].legend()
axs[0].set_title('Model Loss')

axs[1].plot(history_2.history['acc'],label='train_acc')
axs[1].plot(history_2.history['val_acc'],label='val_acc')
axs[1].legend()
axs[1].set_title('Model Accuracy')

plt.show()
plt.close()