import os
from models.zoo.bert.bert_modeling import BertConfig, BertModel
import tensorflow as tf


def new_bert(bert_model, max_seq_length, float_type=tf.float32):
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    valid_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='valid_ids')

    if type(bert_model) == str:
        bert_config = BertConfig.from_json_file(os.path.join(bert_model, "bert_config.json"))
    elif type(bert_model) == dict:
        bert_config = BertConfig.from_dict(bert_model)

    bert_layer = BertModel(config=bert_config, float_type=float_type)
    pooled_output, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)

    return (input_word_ids, input_mask, input_type_ids, valid_ids), (pooled_output, sequence_output), bert_config


def restore_model_ckpt(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(model=model)
    latest_chkpt = tf.train.latest_checkpoint(checkpoint_path)
    checkpoint.restore(latest_chkpt).run_restore_ops()
    return model


def restore_pretrained_weights(model, checkpoint_path, freeze=False):
    model = restore_model_ckpt(model, checkpoint_path)
    if freeze:
        model = restore_model_ckpt(model, checkpoint_path)
        for layer in model.layers:
            if isinstance(layer, BertModel):
                layer.trainable = False
    return model


