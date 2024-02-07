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
from data.util import random_choice_with_seed


# Joined code from two functions to save data as parsing to reduce memory footprint
def parse_and_save_so_file(filename='stackoverflow_0.json', seq_len=12, tokenizer_path='data/ner/vocab.txt', max_client_num=1_000, directory='bert_clients'):
    json_data = load_stackoverflow_json('{}users/{}'.format(DATA_PATH, filename))
    pre_filename = filename.split('.')[0]
    tokenizer = FullTokenizer(tokenizer_path, True)

    j_agents, j_tags = parse_json_agents(json_data)
    process_bert_agents(j_agents, j_tags, seq_len, tokenizer, max_client_num, directory, pre_filename)


def process_bert_agents(j_agents, j_tags, seq_len=128, tokenizer=None, max_client_num=1_000, directory='bert_clients', pre_filename='stackoverflow_0'):
    os.makedirs('data/stackoverflow/{}/'.format(directory), exist_ok=True)

    j_agents_x, part = [], 0

    def save_part():
        save_to_file(j_agents_x, 'data/stackoverflow/{}/clients_{}_{}SL_{}CN_{}PT.h5'.format(directory, pre_filename, seq_len, max_client_num, part))
        return [], part + 1

    for a, t in zip(j_agents, j_tags):
        paragraphs = [clean_text(p) for p in a]
        features = convert_nwp_examples_to_features(paragraphs, seq_len=seq_len, tokenizer=tokenizer)
        j_agents_x.append([features, j_tags])

        if len(j_agents_x) == max_client_num:
            j_agents_x, part = save_part()

    if len(j_agents_x) > 0:
        j_agents_x, part = save_part()


def save_to_file(data_clients, filename):
    hf = h5py.File(filename, 'w')
    for i, agent in enumerate(data_clients):
        c = agent[0]
        num_examples = len(c)
        hf.create_dataset("{}".format(i), data=num_examples)
        hf.create_dataset('{}-input_ids'.format(i), data=[f.input_ids for f in c], dtype=np.int32)
        hf.create_dataset('{}-input_mask'.format(i), data=[f.input_mask for f in c], dtype=np.uint8)
        hf.create_dataset('{}-segment_ids'.format(i), data=[f.segment_ids for f in c], dtype=np.uint8)
        hf.create_dataset('{}-label_id'.format(i), data=[f.label_id for f in c], dtype=np.int32)
        hf.create_dataset('{}-valid_ids'.format(i), data=[f.valid_ids for f in c], dtype=np.uint8)
        hf.create_dataset('{}-label_mask'.format(i), data=[f.label_mask for f in c], dtype=np.uint8)
        hf.create_dataset('{}-tags'.format(i), data=[item for sublist in agent[1] for item in sublist])
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
        tags = hf['{}-tags'.format(i)][:]

        features = [InputFeatures(input_ids=input_ids[j],
                                  input_mask=input_mask[j],
                                  segment_ids=segment_ids[j],
                                  label_id=label_id[j],
                                  valid_ids=valid_ids[j],
                                  label_mask=label_mask[j]) for j in range(num)]
        data_clients.append([features, tags])
        i += 1
    hf.close()
    return data_clients


def load_clients(client_num, seq_len=12, max_client_num=1_000, directory='bert_clients'):
    file_index, part = 0, 0
    clients = []

    def parsed_name():
        return 'data/stackoverflow/{}/clients_stackoverflow_{}_{}SL_{}CN_{}PT.h5'\
            .format(directory, file_index, seq_len, max_client_num, part)

    while len(clients) < client_num or client_num < 0:
        # print("Loading", parsed_name())
        if client_num < 0 and not os.path.exists(parsed_name()):
            return clients
        file_clients = load_from_file(parsed_name())
        part += 1
        if not os.path.exists(parsed_name()):
            file_index += 1
            part = 0
        clients.extend(file_clients)
    return clients


def load_json_client_datasets(num_clients=100, seq_len=12, seed=608361, train_examples_range=(700, 20_000), directory='bert_clients'):
    tokenizer = FullTokenizer('data/ner/vocab.txt', True)

    data = load_stackoverflow_json(f'data/stackoverflow/{directory}/stackoverflow_0.json', verbose=False)
    agents = data['clients_data']

    def train_len(agent):
        return int(sum([len(at.split())-1 for at in agent[0]]) * 0.6)

    choices = [i for i, a in enumerate(agents) if train_examples_range[0] <= train_len(a) <= train_examples_range[1]]
    client_ids = random_choice_with_seed(choices, num_clients, seed)

    agents = [a for i, a in enumerate(agents) if i in client_ids]
    train, val, test, tags = [], [], [], []
    for a, t in agents:
        paragraphs = [clean_text(p) for p in a]
        features = convert_nwp_examples_to_features(paragraphs, seq_len=seq_len, tokenizer=tokenizer)
        f_len = len(features)
        train.append(unpack_features(features[:int(f_len * 0.6)]))
        val.append(unpack_features(features[int(f_len * 0.6): int(f_len * 0.8)]))
        test.append(unpack_features(features[int(f_len * 0.8):]))
        tags.append(t)

    return train, val, test, tags


def load_client_datasets(num_clients=100, seq_len=12, seed=608361, train_examples_range=(700, 20_000), max_client_num=1_000, directory='bert_clients'):
    clients = load_clients(num_clients, seq_len, max_client_num=max_client_num, directory=directory)
    train, val, test, tags = [], [], [], []
    for c_id in range(len(clients)):
        c = clients[c_id][0]
        c_len = len(c)
        train.append(c[:int(c_len * 0.6)])
        val.append(c[int(c_len * 0.6):int(c_len * 0.8)])
        test.append(c[int(c_len * 0.8):])
        tags.append([t.decode() for t in clients[c_id][1]])

    choices = [i for i, tr in enumerate(train) if train_examples_range[0] <= len(tr) <= train_examples_range[1]]
    clients_ids = random_choice_with_seed(choices, num_clients, seed)
    train = [unpack_features(el) for ei, el in enumerate(train) if ei in clients_ids]
    val = [unpack_features(el) for ei, el in enumerate(val) if ei in clients_ids]
    test = [unpack_features(el) for ei, el in enumerate(test) if ei in clients_ids]
    tags = [m for mi, m in enumerate(tags) if mi in clients_ids]
    return train, val, test, tags


def load_clients_data(num_clients=100, seq_len=12, seed=608361, train_examples_range=(700, 20_000)):
    train, val, test, tags = load_client_datasets(num_clients, seq_len, seed, train_examples_range, directory='bert_clients')
    data = {
        "train": train,
        "val": val,
        "test": test,
        "metadata-tags": tags,
        "dataset_name": ['stackoverflow-bert-nwp'] * len(tags),
    }
    return data


if __name__ == '__main__':
    parse_and_save_so_file('stackoverflow_0.json', seq_len=128)
