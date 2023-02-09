import json
import h5py

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
TOKENIZER_PATH = 'data/reddit/reddit_tokenizer.pkl'
DATA_PATH = 'data/reddit/'


def load_reddit_json(json_path):
    with open(json_path, 'r') as inf:
        cdata = json.load(inf)
    print('load_reddit_json', json_path)
    return cdata['user_data']


def read_dir(data_dir='/content', endswith='.json'):
    data = defaultdict(lambda: None)

    if data_dir == '':
        files = os.listdir()
    else:
        files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(endswith)]
    for f in files:
        # print(f)
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])

    return data


def clean_text(text, str_translator):
    text = text.replace('EOS', '').replace('BOS', '').replace('PAD', '').strip() \
        .replace('\n', ' ').replace('\r', ' ').replace('\ufeff', '').translate(str_translator) \
        .replace(' ' * 3, ' ').replace(' ' * 2, ' ').strip()
    return text


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


def text_to_sequence(text_lines, text_tokenizer, max_len):
    seqs = []
    for sentence in text_tokenizer.texts_to_sequences(text_lines):
        for i in range(1, len(sentence)):
            words = sentence[max(i - max_len, 0):i + 1]
            seqs.append(words)

    return np.array(pad_sequences(np.array(seqs), maxlen=max_len + 1, padding='post', value=0))


def parse_json_agents(json_data):
    j_agents_x = []
    j_agents_y = []
    for key in list(json_data.keys()):
        j_agent_data_x = []
        j_agent_data_y = []
        for sx, sy in zip(json_data[str(key)]['x'], json_data[str(key)]['y']):
            s = ' '.join([' '.join(l) for l in sx])
            s = clean_text(s, translator)
            j_agent_data_x.append(s)
            j_agent_data_y.append(sy['subreddit'])

        j_agents_x.append(j_agent_data_x)
        j_agents_y.append(j_agent_data_y)
    return j_agents_x, j_agents_y


def parse_clients(json_data, text_tokenizer, words_backwards, vocab_size, max_client_num, pre_filename):

    j_agents_x, j_agents_y = parse_json_agents(json_data)
    j_clients = []
    for ax, ay in tqdm(zip(j_agents_x, j_agents_y), total=len(j_agents_y)):
        a_seq = text_to_sequence(ax, text_tokenizer, words_backwards)
        # a_x, a_y = a_seq[:, :-1], a_seq[:, -1]
        a_x, a_y = [], []
        for i in range(len(a_seq)):
            ind = np.where(a_seq[i] == 0)
            ind = ind[0][0] - 1 if len(ind[0]) > 0 else len(a_seq[i]) - 1
            a_y.append(a_seq[i][ind])
            a_x.append(np.delete(a_seq[i], ind))
        j_clients.append((np.array(a_x), np.array(a_y), ay))

    prev_ind, cur_ind = 0, max_client_num
    part = 0
    while True:
        save_cli = j_clients[prev_ind:min(cur_ind, len(j_clients))]
        save_to_file(save_cli, 'data/reddit/clients/{}_{}WB_{}VS_{}CN_{}PT.h5'
                     .format(pre_filename, words_backwards, vocab_size, max_client_num, part))
        if len(j_clients) <= cur_ind:
            break
        prev_ind = cur_ind
        cur_ind += max_client_num
        part += 1


def create_tokenizer(vocab_size, train_filenames_list):
    c = Counter()
    for filename in train_filenames_list:
        j_data = load_reddit_json('data/reddit/source/data/reddit_leaf/train/{}'.format(filename))
        agents, _ = parse_json_agents(j_data)
        c.update(' '.join([' '.join(a) for a in agents]).split())

    words = np.array(c.most_common(vocab_size))[:, 0]
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(words)

    pickle.dump(tokenizer, open(TOKENIZER_PATH, 'wb'))
    return tokenizer


def load_tokenizer():
    tokenizer = pickle.load(open(TOKENIZER_PATH, 'rb'))
    return tokenizer


def load_clients(data_type, client_num, word_backwards=10, vocab_size=10_002, max_client_num=1_000):
    reddit_index, part = 0, 0
    clients = []

    def parsed_name():
        return DATA_PATH + 'clients/clients_reddit_{}_{}_{}WB_{}VS_{}CN_{}PT.h5'\
            .format(reddit_index, data_type, word_backwards, vocab_size, max_client_num, part)

    while len(clients) < client_num:
        # print("Loading", parsed_name())
        file_clients = load_from_file(parsed_name())
        part += 1
        if not os.path.exists(parsed_name()):
            reddit_index += 1
            part = 0
        clients.extend(file_clients)
    return clients


def parse_reddit_file(reddit_index=0, word_backwards=10, max_client_num=1_000):
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    # print(vocab_size)
    os.makedirs('data/reddit/clients/', exist_ok=True)
    for data_type in ['train', 'val', 'test']:
        filename = 'reddit_{}_{}.json'.format(reddit_index, data_type)
        print("Parsing", filename)
        json_data = load_reddit_json('data/reddit/source/data/reddit_leaf/{}/{}'.format(data_type, filename))
        parse_clients(json_data, tokenizer,
                      words_backwards=word_backwards,
                      vocab_size=vocab_size,
                      max_client_num=max_client_num,
                      pre_filename='clients_' + filename.split('.')[0])


if __name__ == '__main__':
    # create_tokenizer(10_000, ['reddit_0_train.json'])
    parse_reddit_file(0)
