from data.stackoverflow.preparation import load_stackoverflow_json, parse_json_agents, parse_clients, load_tokenizer, process_agent_data
from data.stackoverflow.clients_data import load_client_datasets, load_json_client_datasets
from data.stackoverflow.bert_clients_data import load_json_client_datasets as load_bert_json_client_datasets, process_bert_agents, FullTokenizer
import numpy as np
import os
from data.util import random_choice_with_seed
from collections import Counter

"""
def parse_bert_per_tag(file_indexes=range(6), seq_len=12, max_client_num=1_000, pct=.9, tags=['python', 'javascript']):
    for d in tags:
        os.makedirs('data/stackoverflow/bert_clients_tag/{}/'.format(d), exist_ok=True)

    tokenizer = FullTokenizer('data/ner/vocab.txt', True)

    agents = {cat: [] for cat in tags}
    for fi in file_indexes:
        json_data = load_stackoverflow_json(f'data/stackoverflow/users/stackoverflow_{fi}.json')
        agent_texts, agents_tags = parse_json_agents(json_data)
        del json_data

        filter_per_tag(agents, agent_texts, agents_tags, pct, tags)

    limit_tag_count(agents)

    for tag_name, tag_agents in agents.items():
        process_bert_agents([el[0] for el in tag_agents], [el[1] for el in tag_agents], seq_len, tokenizer,
                            max_client_num, directory=f'bert_clients_tag/{tag_name}')
"""


def filter_per_tag(agents, agent_texts, agents_tags, pct=.9, tags=['python', 'javascript']):

    for i, tags_y in enumerate(agents_tags):
        m = [tag.split('|')[0] for tag in tags_y]
        tag, count = Counter(m).most_common()[0]
        if tag in tags:
            a_txt, a_tag = agent_texts[i].copy(), agents_tags[i].copy()
            a_txt = [txt for txt, m_tag in zip(a_txt, m) if m_tag == tag]
            a_tag = [t for t, m_tag in zip(a_tag, m) if m_tag == tag]
            agents[tag].append([a_txt, a_tag])
    """
    for i, tags_y in enumerate(agents_tags):
        c_train = [tag in tags_y for tag in tags]
        # exclusively can contain only one tag
        if sum(c_train) == 1:
            tag = np.array(tags)[np.array(c_train)][0]
            if tags_y.count(tag) / len(tags_y) >= pct:
                agents[tag].append([agent_texts[i].copy(), agents_tags[i].copy()])
    # """


def limit_tag_count(agents, max_agents=10_000):
    for k in agents.keys():
        v = agents[k]
        lens = [len(a[1]) for a in v]
        ind = np.argsort(lens)[-max_agents:]
        agents[k] = [el for ei, el in enumerate(v) if ei in ind]


def save_clients_per_tag(file_indexes=range(6), tags=['python', 'javascript']):
    for d in tags:
        os.makedirs('data/stackoverflow/clients_tag/{}/'.format(d), exist_ok=True)

    agents = {cat: [] for cat in tags}
    for fi in file_indexes:
        json_data = load_stackoverflow_json(f'data/stackoverflow/users/stackoverflow_{fi}.json')
        agent_texts, agents_tags = parse_json_agents(json_data)
        del json_data

        filter_per_tag(agents, agent_texts, agents_tags, 1, tags)

    limit_tag_count(agents)

    from models.util import save_json
    for tag_name, tag_agents in agents.items():
        save_json(f'data/stackoverflow/clients_tag/{tag_name}/stackoverflow_0.json', {'clients_data': tag_agents})


"""
def parse_per_tag(file_indexes=range(6), seq_len=10, max_client_num=1_000, pct=.9, tags=['python', 'javascript']):
    for d in tags:
        os.makedirs('data/stackoverflow/clients_tag/{}/'.format(d), exist_ok=True)

    tokenizer = load_tokenizer()

    agents = {cat: [] for cat in tags}
    for fi in file_indexes:
        json_data = load_stackoverflow_json(f'data/stackoverflow/users/stackoverflow_{fi}.json')
        agent_texts, agents_tags = parse_json_agents(json_data)
        del json_data

        filter_per_tag(agents, agent_texts, agents_tags, pct, tags)

    limit_tag_count(agents)

    for tag_name, tag_agents in agents.items():
        process_agent_data([el[0] for el in tag_agents], [el[1] for el in tag_agents], tokenizer, seq_len,
                           len(tokenizer.word_index) + 1, max_client_num,
                           pre_filename=f'clients_stackoverflow_0', directory=f'clients_tag/{tag_name}')
"""


def load_clients_data(num_clients=100, seed=608361, train_examples_range=(700, 20_000), tags=['python', 'javascript'], is_bert=False):
    if not isinstance(num_clients, list) and not isinstance(num_clients, tuple):
        num_clients = int(num_clients / len(tags))
        num_clients = [num_clients] * len(tags)

    data = {
        "train": [],
        "val": [],
        "test": [],
        "metadata-tags": [],
        "dataset_name": []
    }
    for tag, n_cli in zip(tags, num_clients):
        if not is_bert:
            train, val, test, metadata = load_json_client_datasets(num_clients=n_cli, train_examples_range=train_examples_range, seed=seed, directory='clients_tag/{}'.format(tag))
            choices = list(range(len(train)))
            # train, val, test, metadata = load_client_datasets(num_clients=-1, max_client_num=1_000, directory='clients_tag/{}'.format(tag))
            # choices = [i for i, tr in enumerate(train) if train_examples_range[0] <= len(tr[0]) <= train_examples_range[1]]
        else:
            train, val, test, metadata = load_bert_json_client_datasets(num_clients=n_cli, train_examples_range=train_examples_range, seed=seed, directory='clients_tag/{}'.format(tag))
            choices = list(range(len(train)))

        cli_ind = random_choice_with_seed(choices, n_cli, seed)
        data["train"].extend([el for ei, el in enumerate(train) if ei in cli_ind])
        data["val"].extend([el for ei, el in enumerate(val) if ei in cli_ind])
        data["test"].extend([el for ei, el in enumerate(test) if ei in cli_ind])
        metadata_tags = [[t.split('|')[0] for t in el] for ei, el in enumerate(metadata) if ei in cli_ind]
        data["metadata-tags"].extend(metadata_tags)
        dataset_names = ['stackoverflow-nwp-{}'.format(
            np.array(tags)[np.array([t in mt_tags for t in tags])][0])
                         for mt_tags in metadata_tags]
        data["dataset_name"].extend(dataset_names)

    return data


if __name__ == '__main__':
    save_clients_per_tag()
