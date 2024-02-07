from models.abstract_model import *

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, InputLayer
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.optimizers import Adam


def create_model(model_v=0, lr=0.001, decay=0, num_classes=10, input_shape=(32, 32, 3), do_compile=True,
                 default_weights=True, scale=False, center=False, momentum=0.9):
    if model_v == 0:
        model = Sequential(
            [
                Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Flatten(),
                Dense(120, activation='relu'),
                Dense(num_classes, activation='softmax')
            ], "model_{}_{}".format(model_v, next_model_id('cifar10')))
    elif model_v == 100:
        model = Sequential(
                    [
                        Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape),
                        BatchNormalization(momentum=momentum, scale=scale, center=center),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                        Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
                        BatchNormalization(momentum=momentum, scale=scale, center=center),
                        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                        Flatten(),
                        Dense(120, activation='relu'),
                        BatchNormalization(momentum=momentum, scale=scale, center=center),
                        Dense(84, activation='relu'),
                        BatchNormalization(momentum=momentum, scale=scale, center=center),
                        Dense(num_classes, activation='softmax')
                    ], "model_{}_{}".format(model_v, next_model_id('cifar10')))
    elif model_v == 1:
        model = Sequential(
            [
                Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
                BatchNormalization(),
                Conv2D(32, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.3),

                Conv2D(64, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(64, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.5),

                Conv2D(128, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(128, (3, 3), padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.5),

                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ], "model_{}_{}".format(model_v, next_model_id('cifar10')))
    else:
        raise ValueError("Invalid model version")

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights:
        assign_default_weights(model, 'cifar10' + str(model_v))

    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=lr, decay=decay),
                  metrics=[SparseCategoricalAccuracy()])


def load(model_path):
    return load('cifar10/' + model_path, )


def save(model, model_path):
    save(model, 'cifar10/' + model_path)
