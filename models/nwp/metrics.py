import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Precision, Recall
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score


def _get_mask(y_true, sample_weight, masked_tokens):
    if sample_weight is None:
        sample_weight = tf.ones_like(y_true, tf.float32)
    for token in masked_tokens:
        mask = tf.cast(tf.not_equal(y_true, token), tf.float32)
        sample_weight = sample_weight * mask
    return sample_weight


class MaskedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """An accuracy metric that masks some tokens."""

    # noinspection PyDefaultArgument
    def __init__(self, masked_tokens=[1], name='accuracy_no_oov', dtype=None):
        self._masked_tokens = masked_tokens or []
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = _get_mask(y_true, sample_weight, self._masked_tokens)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        mask = tf.reshape(mask, [-1])
        super().update_state(y_true, y_pred, mask)


class MaskedSparseF1Score(F1Score):

    # noinspection PyDefaultArgument
    def __init__(self, masked_tokens=[1], **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'f1_score_no_oov'
        self._masked_tokens = masked_tokens or []
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = _get_mask(y_true, sample_weight, self._masked_tokens)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        mask = tf.reshape(mask, [-1])
        # y_true = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        y_true = tf.one_hot(y_true, num_classes)
        super().update_state(y_true, y_pred, mask)


class MaskedSparsePrecision(Precision):
    # noinspection PyDefaultArgument
    def __init__(self, masked_tokens=[1], **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'sparse_precision_no_oov'
        self._masked_tokens = masked_tokens or []
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = _get_mask(y_true, sample_weight, self._masked_tokens)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        mask = tf.reshape(mask, [-1])
        # y_true = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        y_true = tf.one_hot(y_true, num_classes)
        super().update_state(y_true, y_pred, mask)


class MaskedSparseRecall(Recall):
    # noinspection PyDefaultArgument
    def __init__(self, masked_tokens=[1], **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'sparse_recall_no_oov'
        self._masked_tokens = masked_tokens or []
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = _get_mask(y_true, sample_weight, self._masked_tokens)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        mask = tf.reshape(mask, [-1])
        # y_true = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        y_true = tf.one_hot(y_true, num_classes)
        super().update_state(y_true, y_pred, mask)
