from data.reddit.preparation import process_agent_data, load_reddit_json, parse_json_agents, load_tokenizer
from data.reddit.clients_data import load_client_datasets
from data.reddit.bert_clients_data import load_client_datasets as load_bert_client_datasets
from data.reddit.bert_clients_data import parse_bert_agents, process_bert_agents, FullTokenizer
# from collections import Counter
import numpy as np
import os
from data.util import random_choice_with_seed


def parse_bert_per_subreddit(reddit_indexes=range(21), seq_len=12, max_client_num=1_000, pct=.9, categories=['leagueoflegends', 'politics']):
    for d in categories:
        os.makedirs('data/reddit/bert_clients_category/{}/'.format(d), exist_ok=True)
    tokenizer = FullTokenizer('data/ner/vocab.txt', True)

    agents = {cat: {'train': [], 'val': [], 'test': []} for cat in categories}
    for ri in reddit_indexes:
        train_json_data = load_reddit_json(f'data/reddit/source/data/reddit_leaf/train/reddit_{ri}_train.json')
        train_agents_x, train_agents_y = parse_bert_agents(train_json_data)
        del train_json_data

        val_json_data = load_reddit_json(f'data/reddit/source/data/reddit_leaf/val/reddit_{ri}_val.json')
        val_agents_x, val_agents_y = parse_bert_agents(val_json_data)
        del val_json_data

        test_json_data = load_reddit_json(f'data/reddit/source/data/reddit_leaf/test/reddit_{ri}_test.json')
        test_agents_x, test_agents_y = parse_bert_agents(test_json_data)
        del test_json_data

        filter_per_subreddit(agents, train_agents_x, train_agents_y, val_agents_x, val_agents_y, test_agents_x, test_agents_y, pct, categories)

    for k, v in agents.items():
        for vk, vv in v.items():
            process_bert_agents(j_agents_x=[el[0] for el in vv],
                                j_agents_y=[el[1] for el in vv],
                                seq_len=seq_len, tokenizer=tokenizer, max_client_num=max_client_num,
                                directory=f'bert_clients_category/{k}', pre_filename=f'clients_reddit_0_{vk}')


def filter_per_subreddit(agents, train_agents_x, train_agents_y, val_agents_x, val_agents_y, test_agents_x, test_agents_y, pct=.9, categories=['leagueoflegends', 'politics']):

    for i, (train_y, val_y, test_y) in enumerate(zip(train_agents_y, val_agents_y, test_agents_y)):
        c_train = [cat in train_y for cat in categories]
        # exclusively can contain only one category
        if sum(c_train) == 1:
            # other subsets must not contain other categories
            check_cat = np.array(categories)[np.logical_not(np.array(c_train))]
            s_val = sum([cat in val_y for cat in check_cat])
            s_test = sum([cat in test_y for cat in check_cat])
            if s_val == 0 and s_test == 0:
                # check percentage in all subsets
                cat = np.array(categories)[np.array(c_train)][0]
                if train_y.count(cat) / len(train_y) >= pct and \
                        val_y.count(cat) / len(val_y) >= pct and \
                        test_y.count(cat) / len(test_y) >= pct:
                    agents[cat]['train'].append([train_agents_x[i].copy(), train_agents_y[i].copy()])
                    agents[cat]['val'].append([val_agents_x[i].copy(), val_agents_y[i].copy()])
                    agents[cat]['test'].append([test_agents_x[i].copy(), test_agents_y[i].copy()])


def parse_per_subreddit(reddit_indexes=range(21), seq_len=10, max_client_num=1_000, pct=.9, categories=['leagueoflegends', 'politics']):
    for d in categories:
        os.makedirs('data/reddit/clients_category/{}/'.format(d), exist_ok=True)

    tokenizer = load_tokenizer()

    agents = {cat: {'train': [], 'val': [], 'test': []} for cat in categories}
    for ri in reddit_indexes:

        train_json_data = load_reddit_json(f'data/reddit/source/data/reddit_leaf/train/reddit_{ri}_train.json')
        train_agents_x, train_agents_y = parse_json_agents(train_json_data)
        del train_json_data

        val_json_data = load_reddit_json(f'data/reddit/source/data/reddit_leaf/val/reddit_{ri}_val.json')
        val_agents_x, val_agents_y = parse_json_agents(val_json_data)
        del val_json_data

        test_json_data = load_reddit_json(f'data/reddit/source/data/reddit_leaf/test/reddit_{ri}_test.json')
        test_agents_x, test_agents_y = parse_json_agents(test_json_data)
        del test_json_data

        filter_per_subreddit(agents, train_agents_x, train_agents_y, val_agents_x, val_agents_y, test_agents_x, test_agents_y, pct, categories)

    for k, v in agents.items():
        for vk, vv in v.items():
            process_agent_data([el[0] for el in vv], [el[1] for el in vv], tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename=f'clients_reddit_0_{vk}', directory=f'clients_category/{k}')

    """
    print("Len train", len(train_agents_x), "Len indices:", len(indices))
    train_agents_x, train_agents_y = [train_agents_x[i] for i in indices], [train_agents_y[i] for i in indices]
    val_agents_x, val_agents_y = [val_agents_x[i] for i in indices], [val_agents_y[i] for i in indices]
    test_agents_x, test_agents_y = [test_agents_x[i] for i in indices], [test_agents_y[i] for i in indices]

    c = Counter()
    c.update(' '.join([' '.join(a) for a in train_agents_x]).split())
    words = np.array(c.most_common(vocab_size))[:, 0]
    tokenizer = Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(words)

    process_agent_data(train_agents_x, train_agents_y, tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename='clients_reddit_0_train', directory='clients_category/' + '_'.join(categories))
    process_agent_data(val_agents_x, val_agents_y, tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename='clients_reddit_0_val', directory='clients_category/' + '_'.join(categories))
    process_agent_data(test_agents_x, test_agents_y, tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename='clients_reddit_0_test', directory='clients_category/' + '_'.join(categories))

    for cat in categories:
        indices = [i for i in range(len(train_agents_y)) if cat in train_agents_y[i]]
        train_x, train_y = [train_agents_x[i] for i in indices], [train_agents_y[i] for i in indices]
        
        c = Counter()
        c.update(' '.join([' '.join(a) for a in train_x]).split())
        words = np.array(c.most_common(vocab_size))[:, 0]
        tokenizer = Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(words)
        
        process_agent_data(train_x, train_y, tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename='clients_reddit_0_train', directory='clients_category/{}'.format(cat))
        val_x, val_y = [val_agents_x[i] for i in indices], [val_agents_y[i] for i in indices]
        process_agent_data(val_x, val_y, tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename='clients_reddit_0_val', directory='clients_category/{}'.format(cat))
        test_x, test_y = [test_agents_x[i] for i in indices], [test_agents_y[i] for i in indices]
        process_agent_data(test_x, test_y, tokenizer, seq_len, len(tokenizer.word_index) + 1, max_client_num, pre_filename='clients_reddit_0_test', directory='clients_category/{}'.format(cat))
        """


def load_clients_data(num_clients=100, seed=608361, train_examples_range=(700, 20_000), categories=['leagueoflegends', 'politics'], is_bert=False):
    if not isinstance(num_clients, list) and not isinstance(num_clients, tuple):
        num_clients = int(num_clients / len(categories))
        num_clients = [num_clients] * len(categories)

    data = {
        "train": [],
        "val": [],
        "test": [],
        "metadata-subreddits": [],
        "dataset_name": []
    }
    for cat, n_cli in zip(categories, num_clients):
        if not is_bert:
            train, val, test, metadata = load_client_datasets(num_clients=-1, directory='clients_category/{}'.format(cat))
            choices = [i for i, tr in enumerate(train) if train_examples_range[0] <= len(tr[0]) <= train_examples_range[1]]
        else:
            train, val, test, metadata = load_bert_client_datasets(num_clients=-1, train_examples_range=train_examples_range, seed=-1, directory='bert_clients_category/{}'.format(cat))
            choices = list(range(len(train)))

        cli_ind = random_choice_with_seed(choices, n_cli, seed)
        data["train"].extend([el for ei, el in enumerate(train) if ei in cli_ind])
        data["val"].extend([el for ei, el in enumerate(val) if ei in cli_ind])
        data["test"].extend([el for ei, el in enumerate(test) if ei in cli_ind])
        metadata_subreddits = [el for ei, el in enumerate(metadata) if ei in cli_ind]
        data["metadata-subreddits"].extend(metadata_subreddits)
        dataset_names = ['reddit-nwp-{}'.format(np.array(categories)[np.array([cat in mt_sr for cat in categories])][0])
                         for mt_sr in metadata_subreddits]
        data["dataset_name"].extend(dataset_names)
    return data


if __name__ == '__main__':
    parse_per_subreddit()
