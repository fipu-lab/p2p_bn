# from models import abstract_model as ab_mod
from models.abstract_model import *

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, InputLayer
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.optimizers import Adam


def create_model(model_v=1, lr=0.001, decay=0, num_classes=62, input_shape=(28, 28, 1), do_compile=True,
                 default_weights=True, scale=False, center=False, momentum=0.9):
    layers = [
            InputLayer(input_shape=input_shape),
            Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation="softmax"),
    ]
    if model_v == 2:
        layers.insert(2, BatchNormalization(momentum=momentum, scale=scale, center=center))
        layers.insert(5, BatchNormalization(momentum=momentum, scale=scale, center=center))
        layers.insert(9, BatchNormalization(momentum=momentum, scale=scale, center=center))

    model = Sequential(layers, "model_{}_{}".format(model_v, next_model_id('femnist')))

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights:
        assign_default_weights(model, 'femnist' + str(model_v))
    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=lr, decay=decay),
                  metrics=[SparseCategoricalAccuracy()])


def load(model_path):
    return load('femnist/' + model_path, )


def save(model, model_path):
    save(model, 'femnist/' + model_path)
