import os

from models.zoo.bert.tokenization import FullTokenizer
from models.abstract_model import *

from models.zoo.bert.bert_model import new_bert, restore_pretrained_weights
# from models.zoo.bert.metrics import MaskedSparseCategoricalCrossentropy, MaskedSparseCategoricalAccuracy, MaskedSparseTopKCategoricalAccuracy
MASK_IND = 103


class MaskLayer(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        sequence_output, input_word_ids = inputs[0], inputs[1]
        tmp_indices = tf.where(tf.equal(input_word_ids, tf.constant(MASK_IND, dtype=tf.int32)))  # 103 => Mask index
        mask_words = tf.math.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])

        return tf.boolean_mask(sequence_output, tf.one_hot(mask_words, tf.shape(sequence_output)[1]))


def build_bert_nwp(bert_model, num_labels, max_seq_length):
    (input_word_ids, input_mask, input_type_ids, valid_ids), (pooled_output, sequence_output), bert_config = new_bert(bert_model, max_seq_length)

    sequence_output = MaskLayer()([sequence_output, input_word_ids])
    classifier = tf.keras.layers.Dense(num_labels, activation='softmax', dtype=tf.float32)(sequence_output)

    bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, valid_ids], outputs=[classifier])

    return bert


def create_model(bert_config, seq_len=12, lr=5e-4, decay=0, do_compile=True, default_weights=True):
    bert_path = 'models/zoo/bert/models/' + bert_config
    tokenizer = FullTokenizer(os.path.join(bert_path, "vocab.txt"), True)
    model = build_bert_nwp(bert_path, len(list(tokenizer.vocab.keys())), seq_len)
    model._name = "nwp-bert_{}".format(next_model_id('bert-nwp'))

    if do_compile:
        compile_model(model, lr, decay)

    if type(default_weights) == str:
        if default_weights == 'global':
            # Assign all weights as default
            assign_default_weights(model, 'nwp-bert-' + str(bert_config))
            # Match only bert layer with global weights
            assign_default_weights(model.layers[3], 'global-bert-' + str(bert_config))
        elif 'pretrained' in default_weights:
            model = restore_pretrained_weights(model, bert_path, 'frozen' in default_weights)
        else:
            assign_default_weights(model.layers[3], "{}-{}".format(default_weights, str(bert_config)))
    elif default_weights is True:
        assign_default_weights(model, 'nwp-bert' + str(bert_config))

    return model


"""
# Sequence output
def compile_model(model, lr=0.001, decay=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
        loss=MaskedSparseCategoricalCrossentropy(),
        metrics=[
            MaskedSparseCategoricalAccuracy(),
            MaskedSparseTopKCategoricalAccuracy(k=3, name='masked_sparse_top_3_categorical_accuracy')
        ]
    )
# """


# Pooled output
# """
def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='masked_sparse_top_3_categorical_accuracy')])
# """
