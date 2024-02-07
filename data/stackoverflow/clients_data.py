from data.stackoverflow.preparation import load_clients, load_tokenizer, load_stackoverflow_json, text_to_sequence
import numpy as np
from data.util import random_choice_with_seed


def load_json_client_datasets(num_clients, seq_len=10, seed=608361, train_examples_range=(700, 20_000), directory='clients'):
    data = load_stackoverflow_json(f'data/stackoverflow/{directory}/stackoverflow_0.json', verbose=False)
    agents = data['clients_data']

    def train_len(agent):
        return int(sum([len(at.split())-1 for at in agent[0]]) * 0.6)

    choices = [i for i, a in enumerate(agents) if train_examples_range[0] <= train_len(a) <= train_examples_range[1]]
    client_ids = random_choice_with_seed(choices, num_clients, seed)

    agents = [a for i, a in enumerate(agents) if i in client_ids]
    text_tokenizer = load_tokenizer()
    train, val, test, tags = [], [], [], []
    for a_text, a_tags in agents:
        a_seq = text_to_sequence(a_text, text_tokenizer, seq_len)
        a_x, a_y = [], []
        for i in range(len(a_seq)):
            ind = np.where(a_seq[i] == 0)
            ind = ind[0][0] - 1 if len(ind[0]) > 0 else len(a_seq[i]) - 1
            a_y.append(a_seq[i][ind])
            a_x.append(np.delete(a_seq[i], ind))

        x = np.array(a_x)
        y = np.array(a_y)
        a_len = len(y)
        train.append([x[:int(a_len * 0.6)], y[:int(a_len * 0.6)]])
        val.append([x[int(a_len * 0.6):int(a_len * 0.8)], y[int(a_len * 0.6):int(a_len * 0.8)]])
        test.append([x[int(a_len * 0.8):], y[int(a_len * 0.8):]])
        tags.append(a_tags)
    return train, val, test, tags


def load_client_datasets(num_clients, max_client_num=10_000, directory='clients'):
    clients = load_clients(num_clients, max_client_num=max_client_num, directory=directory)
    train, val, test, tags = [], [], [], []
    for c_id in range(len(clients)):
        c = clients[c_id]
        c_len = len(c[1])
        train.append([c[0][:int(c_len * 0.6)], c[1][:int(c_len * 0.6)]])
        val.append([c[0][int(c_len * 0.6):int(c_len * 0.8)], c[1][int(c_len * 0.6):int(c_len * 0.8)]])
        test.append([c[0][int(c_len * 0.8):], c[1][int(c_len * 0.8):]])
        tags.append([item.decode() for item in c[2]])
    return train, val, test, tags


def load_clients_data(num_clients=100, seed=608361, train_examples_range=(700, 20_000), max_client_num=10_000, directory='clients'):
    train, val, test, tags = load_client_datasets(25_000, max_client_num=max_client_num, directory=directory)
    choices = [i for i, tr in enumerate(train) if train_examples_range[0] < len(tr[1]) <= train_examples_range[1]]
    client_ids = random_choice_with_seed(choices, num_clients, seed)
    data = {
        "train": [el for ei, el in enumerate(train) if ei in client_ids],
        "val": [el for ei, el in enumerate(val) if ei in client_ids],
        "test": [el for ei, el in enumerate(test) if ei in client_ids],
        "metadata-tags": [el for ei, el in enumerate(tags) if ei in client_ids],
        "dataset_name": ['stackoverflow-nwp'] * num_clients,
    }
    return data
