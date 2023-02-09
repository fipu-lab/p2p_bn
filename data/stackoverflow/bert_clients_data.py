import string
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from models.zoo.bert.tokenization import FullTokenizer
import h5py

from data.reddit.bert_clients_data import convert_nwp_examples_to_features, clean_text, unpack_features, InputFeatures
from data.stackoverflow.preparation import load_stackoverflow_json, DATA_PATH, parse_json_agents


# Joined code from two functions to save data as parsing to reduce memory footprint
def parse_and_save_so_file(filename='stackoverflow_0.json', seq_len=12, tokenizer_path='data/ner/vocab.txt', max_client_num=1_000):
    json_data = load_stackoverflow_json('{}users/{}'.format(DATA_PATH, filename))
    os.makedirs('data/stackoverflow/bert_clients/', exist_ok=True)
    pre_filename = filename.split('.')[0]
    tokenizer = FullTokenizer(tokenizer_path, True)

    j_agents = parse_json_agents(json_data)

    j_agents_x, part = [], 0

    def save_part():
        save_to_file(j_agents_x, 'data/stackoverflow/bert_clients/clients_{}_{}SL_{}CN_{}PT.h5'.format(pre_filename, seq_len, max_client_num, part))
        return [], part + 1

    for a in j_agents:
        paragraphs = [clean_text(p) for p in a]
        features = convert_nwp_examples_to_features(paragraphs, seq_len=seq_len, tokenizer=tokenizer)
        j_agents_x.append(features)

        if len(j_agents_x) == max_client_num:
            j_agents_x, part = save_part()

    if len(j_agents_x) > 0:
        j_agents_x, part = save_part()


def save_to_file(data_clients, filename):
    hf = h5py.File(filename, 'w')
    for i, c in enumerate(data_clients):
        num_examples = len(c)
        hf.create_dataset("{}".format(i), data=num_examples)
        hf.create_dataset('{}-input_ids'.format(i), data=[f.input_ids for f in c], dtype=np.int32)
        hf.create_dataset('{}-input_mask'.format(i), data=[f.input_mask for f in c], dtype=np.uint8)
        hf.create_dataset('{}-segment_ids'.format(i), data=[f.segment_ids for f in c], dtype=np.uint8)
        hf.create_dataset('{}-label_id'.format(i), data=[f.label_id for f in c], dtype=np.int32)
        hf.create_dataset('{}-valid_ids'.format(i), data=[f.valid_ids for f in c], dtype=np.uint8)
        hf.create_dataset('{}-label_mask'.format(i), data=[f.label_mask for f in c], dtype=np.uint8)
    hf.close()


def load_from_file(filename):
    data_clients = []
    hf = h5py.File(filename, 'r')
    i = 0
    while hf.get("{}".format(i)):
        num = hf.get("{}".format(i))[()]
        input_ids = hf['{}-input_ids'.format(i)][:]
        input_mask = hf['{}-input_mask'.format(i)][:]
        segment_ids = hf['{}-segment_ids'.format(i)][:]
        label_id = hf['{}-label_id'.format(i)][:]
        valid_ids = hf['{}-valid_ids'.format(i)][:]
        label_mask = hf['{}-label_mask'.format(i)][:]

        features = [InputFeatures(input_ids=input_ids[j],
                                  input_mask=input_mask[j],
                                  segment_ids=segment_ids[j],
                                  label_id=label_id[j],
                                  valid_ids=valid_ids[j],
                                  label_mask=label_mask[j]) for j in range(num)]
        data_clients.append(features)
        i += 1
    hf.close()
    return data_clients


def load_clients(client_num, seq_len=12, max_client_num=1_000):
    file_index, part = 0, 0
    clients = []

    def parsed_name():
        return 'data/stackoverflow/bert_clients/clients_stackoverflow_{}_{}SL_{}CN_{}PT.h5'\
            .format(file_index, seq_len, max_client_num, part)

    while len(clients) < client_num:
        # print("Loading", parsed_name())
        file_clients = load_from_file(parsed_name())
        part += 1
        if not os.path.exists(parsed_name()):
            file_index += 1
            part = 0
        clients.extend(file_clients)
    return clients


def load_clients_data(num_clients=100, seq_len=12, seed=608361, train_examples_range=(700, 20_000)):
    clients = load_clients(num_clients, seq_len)
    choices = [i for i, tr in enumerate(clients) if train_examples_range[0] <= len(tr) <= train_examples_range[1]]
    if seed is not None:
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        rs = RandomState(MT19937(SeedSequence(seed)))
        clients_ids = rs.choice(choices, size=num_clients, replace=False)
    else:
        clients_ids = np.random.choice(choices, size=num_clients, replace=False)

    train, val, test = [], [], []
    for c_id in clients_ids:
        c = clients[c_id]
        c_len = len(c)
        train.append(c[:int(c_len * 0.6)])
        val.append(c[int(c_len * 0.6):int(c_len * 0.8)])
        test.append(c[int(c_len * 0.8):])

    data = {
        "train": [unpack_features(el) for el in train],
        "val": [unpack_features(el) for el in val],
        "test": [unpack_features(el) for el in test],
        "dataset_name": ['stackoverflow-bert-nwp'] * num_clients,
    }
    return data


if __name__ == '__main__':
    parse_and_save_so_file('stackoverflow_0.json', seq_len=128)
