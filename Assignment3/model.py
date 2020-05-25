from tensorflow import keras
from tensorflow.keras import layers


def classifier(inpt, output_dim):
    model = keras.Sequential()
    model.add(keras.Input(shape=inpt))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(output_dim, activation='softmax'))

    return model
