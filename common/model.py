import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import KLDivergence

from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import clone_model

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from common.metrics import MaskedSparseCategoricalAccuracy, MaskedSparseF1Score, MaskedSparsePrecision, MaskedSparseRecall
from tensorflow.keras.metrics import Recall
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

import numpy as np
import gc

_MODEL_COUNT = 0

_default_weights = {
}


def clear_def_weights_cache():
    _default_weights.clear()


def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=lr, decay=decay),
                  metrics=[MaskedSparseCategoricalAccuracy()])


def create_model(model_v=1, lr=0.001, decay=0, vocab_size=10002, embedding_size=10, do_compile=True, default_weights=False):
    global _MODEL_COUNT
    units = 25 if model_v % 2 == 1 else 100
    use_bn = model_v > 2

    layers = [Embedding(vocab_size, units, input_length=embedding_size, trainable=True, mask_zero=True),
              GRU(units)]

    if use_bn:
        layers.append(BatchNormalization(momentum=0.9, scale=False, center=False))
    layers.append(Dense(units))

    if use_bn:
        layers.append(BatchNormalization(momentum=0.9, scale=False, center=False))
    layers.append(Dense(vocab_size, activation='softmax'))

    model = Sequential(layers, "model_{}_{}".format(model_v, _MODEL_COUNT))

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights:
        if model_v in _default_weights:
            model.set_weights(_default_weights[model_v])
        else:
            _default_weights[model_v] = model.get_weights()

    _MODEL_COUNT += 1
    return model


def reset_compiled_metrics(model):
    model.compiled_metrics.reset_state()


def eval_model_metrics(m, dataset):
    if len(m.compiled_metrics.metrics) == 0:
        m.compiled_metrics.build(0, 0)
    metrics = m.compiled_metrics.metrics
    for metric in metrics:
        if hasattr(metric, "reset_state"):
            metric.reset_state()
        else:
            metric.reset_states()
    for (dx, dy) in dataset:
        preds = m(dx, training=False)
        for metric in metrics:
            metric.update_state(dy, preds)
    return {metric.name: metric.result().numpy() for metric in metrics}


def calculate_memory_model_size(model):
    return model.count_params() * 4 / (1024 ** 2) * 2


def load(model_path):
    if not model_path.endswith('.h5'):
        model_path += '.h5'
    model = load_model('models/' + model_path,
                       custom_objects={"MaskedSparseCategoricalAccuracy": MaskedSparseCategoricalAccuracy()})
    return model


def save(model, model_path, signatures=None):
    if not model_path.endswith('.h5'):
        model_path += '.h5'
    save_model(model, 'models/' + model_path, save_format='h5', signatures=signatures)
    clear_session()
    # gc.collect()
    # del model


def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()


def load_tf(model_path):
    return tf.saved_model.load('models/' + model_path)


def save_tf(model, model_path, signatures=None):
    tf.saved_model.save(model, 'models/' + model_path, signatures=signatures)


def average_weights(model_weights):
    return __apply_on_weights(model_weights, np.average)


def __apply_on_weights(model_weights, np_func):
    weights = []

    # determine how many layers need to be averaged
    n_layers = len(model_weights[0])
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = np.array([m_weight[layer] for m_weight in model_weights])
        # weighted average of weights for this layer
        avg_layer_weights = np_func(layer_weights, axis=0)

        weights.append(avg_layer_weights)

    return weights


@tf.function
def kl_loss_compute(logits1, logits2):
    """ KL loss
    """
    pred1 = tf.math.softmax(logits1)
    pred2 = tf.math.softmax(logits2)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(pred2 * tf.math.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss


def multiply_weights_with_num(weights, num):
    for i in range(len(weights)):
        weights[i] *= num
    return weights


def add_weights(model_weights):
    return __apply_on_weights(model_weights, np.sum)


"""
def avg_weights(weight_list):
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad
"""


def average_trainable_variables(trainable_variable_list):
    n_vars = len(trainable_variable_list[0])
    train_vars = []
    for n in range(n_vars):
        n_var_list = np.array([tra_var[n].numpy() for tra_var in trainable_variable_list])
        avg_n_var = np.mean(n_var_list, axis=0)
        train_vars.append(avg_n_var)
    return train_vars
