import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

data = pd.read_csv('/content/x-flow-team/source/XrayFeature_train.csv').values
name = data[:, 1]
feature = data[:, 2:].astype('float32')

data_label = pd.read_csv('/content/x-flow-team/source/train.csv', usecols=[0,4]).values
label_dict = {}
for i in range(data_label.shape[0]):
  label_dict[data_label[i][0]] = data_label[i][1]
label = np.empty((name.shape[0],1), dtype='float32')
for index, item in enumerate(name):
  label[index] = label_dict[item]
all_data = np.concatenate((feature, label), axis=1)
np.random.shuffle(all_data)
train_number = int(all_data.shape[0]*0.8)
train_data, train_label = all_data[:train_number, :-1], all_data[:train_number, -1]
val_data, val_label = all_data[train_number:, :-1], all_data[train_number:, -1]

model = models.Sequential()
model.add(layers.Flatten(input_shape=(1,1024)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, train_label, epochs=20, 
                    validation_data=(val_data, val_label))
model.save('/content/x-flow-team/source/saved_model/my_model')

plt.figure()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()