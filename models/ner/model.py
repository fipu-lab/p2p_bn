import numpy as np

from models.abstract_model import *
from models.zoo.bert.metrics import MaskedSparseCategoricalCrossentropy
from seqeval.metrics import classification_report, accuracy_score

from data.ner.dataset_loader import CoNLLProcessor, FewNERDProcessor
from models.zoo.bert.bert_model import new_bert, restore_pretrained_weights

ner_processors = {
    'conll': CoNLLProcessor('data/ner/conll'),
    'few': FewNERDProcessor('data/ner/few_nerd')
}


class ValidationLayer(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        # Function that removes invalid tokens (such as #es) and adds padding to the end
        def val_fn(x):
            valid = tf.boolean_mask(x[0], x[1], axis=0)
            v_shape = tf.shape(valid)
            zeros = tf.cast(tf.fill((tf.math.subtract(tf.shape(x[0])[0], v_shape[0]), v_shape[1]), 0), dtype=tf.float32)
            return tf.concat([valid, zeros], axis=0)

        sequence_output, valid_ids = inputs[0], inputs[1]
        valid_ids = tf.cast(valid_ids, tf.bool)
        return tf.map_fn(val_fn, (sequence_output, valid_ids),
                         dtype=(tf.float32, tf.bool), fn_output_signature=tf.float32)


def build_bert_ner(bert_model, num_labels, max_seq_length):
    (input_word_ids, input_mask, input_type_ids, valid_ids), (pooled_output, sequence_output), bert_config = new_bert(bert_model, max_seq_length)

    val_layer = ValidationLayer()([sequence_output, valid_ids])
    # dropout = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(val_layer)
    initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)

    classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=tf.float32)(val_layer)

    bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, valid_ids], outputs=[classifier])

    return bert


def create_model(bert_config, processor_name='conll', seq_len=128, lr=5e-4, decay=0, do_compile=True, default_weights=True):
    bert_path = 'models/zoo/bert/models/' + bert_config
    processor = ner_processors[processor_name]
    model = build_bert_ner(bert_path, processor.label_len(), seq_len)

    model._name = "ner_{}_{}".format(processor_name, next_model_id('ner_{}'.format(processor_name)))

    if do_compile:
        compile_model(model, lr, decay)

    if type(default_weights) == str:
        if default_weights == 'global':
            # Assign all weights as default
            assign_default_weights(model, 'bert-ner-' + str(bert_config))
            # Match only bert layer with global weights
            assign_default_weights(model.layers[3], 'global-bert-' + str(bert_config))
        elif 'pretrained' in default_weights:
            model = restore_pretrained_weights(model, bert_path, 'frozen' in default_weights)
        else:
            assign_default_weights(model.layers[3], "{}-{}".format(default_weights, str(bert_config)))
    elif default_weights is True:
        assign_default_weights(model, 'bert-ner-{}-{}'.format(processor_name, str(bert_config)))

    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
        loss=MaskedSparseCategoricalCrossentropy()
    )


def evaluate(model, batched_eval_data, label_map, out_ind, sep_ind, pad_ind, do_print=False):
    y_true, y_pred = [], []

    def label_map_fn(x):
        return label_map[x] if label_map[x] not in ['[SEP]', '[PAD]', '[CLS]'] else label_map[out_ind]

    for (input_ids, input_masks, segment_ids, valid_ids), (label_ids, label_mask) in batched_eval_data:
        logits = model((input_ids, input_masks, segment_ids, valid_ids), training=False)
        logits = tf.argmax(logits, axis=2)
        for i in range(label_ids.shape[0]):
            lbl_ids = label_ids[i].numpy()
            lbl_ids = lbl_ids[lbl_ids > pad_ind]  # Skipping padding when evaluating

            sep_indxs = np.where(lbl_ids == sep_ind)[0]
            lbl_ids = np.split(lbl_ids, sep_indxs)[0]
            pred_ids = np.split(logits[i].numpy(), sep_indxs)[0]

            y_true.append([label_map_fn(x) for x in lbl_ids])
            y_pred.append([label_map_fn(x) for x in pred_ids])

    return _do_classification_report(y_true, y_pred, label_map, do_print)


def eval_model_metrics(m, dataset):
    processor = ner_processors[m.name.split('_')[1]]
    return evaluate(m, dataset,
                    processor.get_label_map(),
                    processor.token_ind('O'),
                    processor.token_ind('[SEP]'),
                    processor.token_ind('[PAD]'))


def eval_ensemble_metrics(models, dataset, metrics, weights=None):
    if weights is None:
        weights = len(models) * [1 / len(models)]
    # TODO: it is possible that different models have different processors
    processor = ner_processors[models[0].name.split('_')[1]]
    return evaluate_models(models, weights, dataset,
                           processor.get_label_map(),
                           processor.token_ind('O'),
                           processor.token_ind('[SEP]'),
                           processor.token_ind('[PAD]'))


def evaluate_models(models, weights, batched_eval_data, label_map, out_ind, sep_ind, pad_ind, do_print=False):
    y_true, y_pred = [], []

    def label_map_fn(x):
        return label_map[x] if label_map[x] not in ['[SEP]', '[PAD]', '[CLS]'] else label_map[out_ind]

    for (input_ids, input_masks, segment_ids, valid_ids), (label_ids, label_mask) in batched_eval_data:
        logits_list = [model((input_ids, input_masks, segment_ids, valid_ids), training=False) for model in models]
        logits = sum((m_logits * w for m_logits, w in zip(logits_list, weights)))
        logits = tf.argmax(logits, axis=2)
        for i in range(label_ids.shape[0]):
            lbl_ids = label_ids[i].numpy()
            lbl_ids = lbl_ids[lbl_ids > pad_ind]  # Skipping padding when evaluating

            sep_indxs = np.where(lbl_ids == sep_ind)[0]
            lbl_ids = np.split(lbl_ids, sep_indxs)[0]
            pred_ids = np.split(logits[i].numpy(), sep_indxs)[0]

            y_true.append([label_map_fn(x) for x in lbl_ids])
            y_pred.append([label_map_fn(x) for x in pred_ids])

    return _do_classification_report(y_true, y_pred, label_map, do_print)


def _do_classification_report(y_true, y_pred, label_map, do_print=False):
    if do_print:
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    class_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    entities = np.unique(
        [val.replace('B-', '').replace('I-', '') for val in label_map.values() if '[' not in val and val != 'O'])
    for entity in entities:
        if entity not in class_dict:
            class_dict[entity] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    out_dict = {}
    for k, v in class_dict.items():
        for dk, dv in v.items():
            out_dict[k.replace('-', '_').replace(' ', '_') + '_' + dk.replace('-', '_').replace(' ', '_')] = dv
    out_dict['ner_accuracy'] = accuracy_score(y_true, y_pred)
    return out_dict
