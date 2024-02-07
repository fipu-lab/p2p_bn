import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import gc
import numpy as np

_MODEL_COUNT = {}

_default_weights = {}


def next_model_id(name):
    if name not in _MODEL_COUNT:
        _MODEL_COUNT[name] = 0
    _MODEL_COUNT[name] += 1
    return _MODEL_COUNT[name] - 1


def assign_default_weights(model, name):
    if name in _default_weights:
        model.set_weights(_default_weights[name])
    else:
        _default_weights[name] = model.get_weights()


def clear_def_weights_cache():
    _default_weights.clear()


def compile_model(model, **kwargs):
    pass


def create_model(**kwargs):
    pass


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


def eval_model_loss(m, dataset):
    total_loss = 0
    num_batches = 0
    for x, y in dataset:
        logits = m(x, training=False)
        loss = m.loss(y, logits)
        total_loss += loss
        num_batches += 1
    average_loss = total_loss / num_batches
    return average_loss.numpy()


def eval_ensemble_metrics(models, dataset, metrics, weights=None):
    if weights is None:
        weights = len(models) * [1 / len(models)]

    for metric in metrics:
        if hasattr(metric, "reset_state"):
            metric.reset_state()
        else:
            metric.reset_states()

    for (dx, dy) in dataset:
        m_preds = [m(dx, training=False) for m in models]
        for metric in metrics:
            metric.update_state(dy, sum((pred * w for pred, w in zip(m_preds, weights))))
    return {metric.name: metric.result().numpy() for metric in metrics}


def calculate_memory_model_size(model):
    return model.count_params() * 4 / (1024 ** 2) * 2


def load(model_path, custom_objects=None):
    if not model_path.endswith('.h5'):
        model_path += '.h5'
    model = load_model('log/models/' + model_path, custom_objects=custom_objects)
    return model


def save(model, model_path, signatures=None):
    if not model_path.endswith('.h5'):
        model_path += '.h5'
    save_model(model, 'log/models/' + model_path, save_format='h5', signatures=signatures)
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


@tf.function
def kl_loss_compute(logits1, logits2):
    """ KL loss
    """
    pred1 = tf.math.softmax(logits1)
    pred2 = tf.math.softmax(logits2)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(pred2 * tf.math.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss


def weights_average(weights, alphas=None):
    if alphas is None:
        alphas = [1 / len(weights)] * len(weights)
    else:
        alphas = np.array(alphas)/np.sum(alphas)
    new_weights = []

    for l_i in range(len(weights[0])):
        avg_layer = tf.convert_to_tensor(np.sum([w[l_i]*a for w, a in zip(weights, alphas)], axis=0))
        new_weights.append(avg_layer)

    return new_weights
