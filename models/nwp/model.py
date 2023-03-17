# from models import abstract_model as ab_mod
from models.abstract_model import *

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

from models.nwp.metrics import MaskedSparseCategoricalAccuracy


def create_model(model_v=1, lr=0.001, decay=0, vocab_size=10002, embedding_size=10, do_compile=True, default_weights=True):
    units = 25 if model_v % 2 == 1 else 100
    use_bn = model_v > 2

    model_layers = [Embedding(vocab_size, units, input_length=embedding_size, trainable=True, mask_zero=True),
                    GRU(units)]

    if use_bn:
        model_layers.append(BatchNormalization(momentum=0.9, scale=False, center=False))
    model_layers.append(Dense(units))

    if use_bn:
        model_layers.append(BatchNormalization(momentum=0.9, scale=False, center=False))
    model_layers.append(Dense(vocab_size, activation='softmax'))

    model = Sequential(model_layers, "model_{}_{}".format(model_v, next_model_id('nwp')))

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights:
        assign_default_weights(model, 'nwp' + str(model_v))

    compile_model(model, lr, decay)
    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=lr, decay=decay),
                  metrics=[MaskedSparseCategoricalAccuracy()])


def load(model_path):
    return load('nwp/' + model_path, custom_objects={"MaskedSparseCategoricalAccuracy": MaskedSparseCategoricalAccuracy()})


def save(model, model_path):
    save(model, 'nwp/' + model_path)
