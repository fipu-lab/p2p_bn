import json
import h5py
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import string
from collections import Counter

translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
TOKENIZER_PATH = 'data/stackoverflow/stackoverflow_tokenizer.pkl'
DATA_PATH = 'data/stackoverflow/'


def load_stackoverflow_json(json_path):
    with open(json_path, 'r') as inf:
        cdata = json.load(inf)
    print('load_stackoverflow_json', json_path)
    return cdata


def save_to_file(data_clients, filename):
    hf = h5py.File(filename, 'w')
    for i, c in enumerate(data_clients):
        hf.create_dataset("{}".format(i), data=len(c))
        for j, s in enumerate(c):
            hf.create_dataset("{}-{}".format(i, j), data=s)
    hf.close()


def load_from_file(filename):
    data_clients = []
    hf = h5py.File(filename, 'r')
    i = 0
    while hf.get("{}".format(i)):
        num = hf.get("{}".format(i))[()]
        sentences = []
        for j in range(num):
            s = hf.get("{}-{}".format(i, j))[()]
            sentences.append(s)
        data_clients.append(sentences)
        i += 1
    hf.close()
    return data_clients


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """
    # text = re.sub(r"\[(.+)\]\(.+\)", r"\1", markdown_string)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", markdown_string)

    # text = re.sub(r"\*\*(?=)", r"", text)
    text = re.sub(r"\*(?=)", r"", text)

    # if markdown_string != text:
    #     print("---BEFORE ---", markdown_string)
    #     print("---AFTER  ---", text)
    return text


def clean_text(text, str_translator):
    text = text.lower()
    text = markdown_to_text(text)
    text = text.strip() \
        .replace('\n', ' ').replace('\r', ' ').replace('\ufeff', '').translate(str_translator) \
        .replace(' ' * 3, ' ').replace(' ' * 2, ' ').strip()
    return text


def parse_json_agents(json_data):
    j_agents = []
    for u_id, u_comms in json_data.items():
        j_agents.append([clean_text(t, translator) for t in u_comms if t is not None])
    return j_agents


def text_to_sequence(text_lines, text_tokenizer, max_len):
    seqs = []
    for sentence in text_tokenizer.texts_to_sequences(text_lines):
        for i in range(1, len(sentence)):
            words = sentence[max(i - max_len, 0):i + 1]
            seqs.append(words)

    return np.array(pad_sequences(np.array(seqs), maxlen=max_len + 1, padding='pre'))

# data = load_stackoverflow_json(DATA_PATH + 'stackoverflow_0.json')


def parse_clients(json_data, text_tokenizer, words_backwards, vocab_size, max_client_num, pre_filename):

    j_agents = parse_json_agents(json_data)
    j_clients = []
    for a in tqdm(j_agents):
        a_seq = text_to_sequence(a, text_tokenizer, words_backwards)
        a_x, a_y = a_seq[:, :-1], a_seq[:, -1]
        j_clients.append((np.array(a_x), np.array(a_y)))

    prev_ind, cur_ind = 0, max_client_num
    part = 0
    while True:
        save_cli = j_clients[prev_ind:min(cur_ind, len(j_clients))]
        save_to_file(save_cli, DATA_PATH + 'clients/{}_{}WB_{}VS_{}CN_{}PT.h5'
                     .format(pre_filename, words_backwards, vocab_size, max_client_num, part))
        if len(j_clients) <= cur_ind:
            break
        prev_ind = cur_ind
        cur_ind += max_client_num
        part += 1


def create_tokenizer(vocab_size, train_filenames_list):
    c = Counter()
    for filename in train_filenames_list:
        j_data = load_stackoverflow_json('{}users/{}'.format(DATA_PATH, filename))
        agents = parse_json_agents(j_data)
        c.update(''.join([''.join(a) for a in agents]).split())

    # print(c.most_common(vocab_size))
    words = np.array(c.most_common(vocab_size))[:, 0]
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(words)

    pickle.dump(tokenizer, open(TOKENIZER_PATH, 'wb'))
    return tokenizer


def load_tokenizer():
    tokenizer = pickle.load(open(TOKENIZER_PATH, 'rb'))
    return tokenizer


def load_clients(client_num, word_backwards=10, vocab_size=10_002, max_client_num=10_000):
    stackoverflow_index, part = 0, 0
    clients = []

    def parsed_name():
        return DATA_PATH + 'clients/clients_stackoverflow_{}_{}WB_{}VS_{}CN_{}PT.h5'\
            .format(stackoverflow_index, word_backwards, vocab_size, max_client_num, part)

    while len(clients) < client_num:
        # print("Loading", parsed_name())
        file_clients = load_from_file(parsed_name())
        part += 1
        if not os.path.exists(parsed_name()):
            stackoverflow_index += 1
            part = 0
        clients.extend(file_clients)
    return clients


def parse_stackoverflow_file(stackoverflow_index=0, word_backwards=10, max_client_num=1_000):
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    # print(vocab_size)
    filename = 'stackoverflow_{}.json'.format(stackoverflow_index)
    print("Parsing", filename)
    json_data = load_stackoverflow_json('{}users/{}'.format(DATA_PATH, filename))
    parse_clients(json_data, tokenizer,
                  words_backwards=word_backwards,
                  vocab_size=vocab_size,
                  max_client_num=max_client_num,
                  pre_filename='clients_' + filename.split('.')[0])


if __name__ == '__main__':
    # create_tokenizer(10_000, ['stackoverflow_0.json'])
    parse_stackoverflow_file(0, max_client_num=10_000)
