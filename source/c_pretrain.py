from os import name
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from misc_tools import relative_path

# 
# load
#
def load_data():
    data = pd.read_csv(relative_path('XrayFeature_train.csv')).values
    name = data[:, 1]
    feature = data[:, 2:].astype('float32')

    data_label = pd.read_csv(relative_path('train.csv'), usecols=["filename","covid(label)"]).values
    label_dict = {} # = { "img_93.jpg": 1, "img_94.jpg": 0, ... }
    for i in range(data_label.shape[0]):
        label_dict[data_label[i][0]] = data_label[i][1]

    label = np.empty((name.shape[0],1), dtype='float32')
    for index, item in enumerate(name):
        label[index] = label_dict[item]
    all_data = np.concatenate((feature, label), axis=1)
    return all_data

# 
# train/validation split
#
def get_train_validate_split(all_data):
    np.random.shuffle(all_data)
    train_number = int(all_data.shape[0]*0.8)
    train_data, train_label = all_data[:train_number, :-1], all_data[:train_number, -1]
    val_data, val_label = all_data[train_number:, :-1], all_data[train_number:, -1]
    return (train_data, train_label), (val_data, val_label)

# 
# create model
#
def create_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(1,1024)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# 
# train model
# 
def train_model(model, train_data, train_label, val_data, val_label):
    history = model.fit(
        train_data,
        train_label,
        epochs=20, 
        validation_data=(val_data, val_label)
    )
    model.save(relative_path('pretrain.model'))
    return history

# 
# Accuracy
# 
def plot_accuracy(history):
    figure = plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    # plt.show()
    image_path = relative_path("../graphs/nn_model_accuracy")
    figure.savefig(image_path)
    print(f"saved image: {image_path}")

# 
# Loss
# 
def plot_loss(history):
    figure = plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    # plt.show()
    image_path = relative_path("../graphs/nn_model_loss")
    figure.savefig(image_path)
    print(f"saved image: {image_path}")

if __name__ == "__main__":
    all_data = load_data()
    (train_data, train_label), (val_data, val_label) = get_train_validate_split(all_data)
    model = create_model()
    history = train_model(model, train_data, train_label, val_data, val_label)
    plot_accuracy(history)
    plot_loss(history)