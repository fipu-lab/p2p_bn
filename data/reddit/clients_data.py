from data.reddit.preparation import load_clients
import numpy as np


def load_client_datasets(num_clients=1_000):
    train = load_clients('train', num_clients)
    val = load_clients('val', num_clients)
    test = load_clients('test', num_clients)
    metadata = []
    for tr, v, ts in zip(train, val, test):
        subreddits = [d.decode() for d in tr[2]] + [d.decode() for d in v[2]] + [d.decode() for d in ts[2]]
        metadata.append(np.unique(subreddits))

    train = [el[:2] for el in train]
    val = [el[:2] for el in val]
    test = [el[:2] for el in test]
    return train, val, test, metadata


def load_clients_data(num_clients=100, starting_client=0):
    tr, val, test, metadata = load_client_datasets(num_clients + starting_client)
    data = {
        "train": tr[starting_client:num_clients + starting_client],
        "val": val[starting_client:num_clients + starting_client],
        "test": test[starting_client:num_clients + starting_client],
        "metadata-subreddits": metadata[starting_client:num_clients + starting_client],
        "dataset_name": ['reddit-nwp'] * num_clients,
    }
    return data


def filter_clients(train_clients, val_clients, test_clients, cli_num, examples_range, ret_val='clients'):
    min_examples, max_examples = examples_range[0], examples_range[1]
    print("{} Clients, size: {} - {}".format(cli_num, min_examples, max_examples))
    c_added = 0
    cls = list(range(len(train_clients)))
    cl_ds = []
    while c_added < cli_num and len(cls) > 0:
        while len(train_clients[cls[0]][0]) < min_examples or len(train_clients[cls[0]][0]) > max_examples:
            cls.pop(0)
        c_added += 1
        k = cls[0]
        cl_ds.append(k)
        cls.pop(0)
    if ret_val == 'clients':
        return [train_clients[d] for d in cl_ds], \
               [val_clients[d] for d in cl_ds], \
               [test_clients[d] for d in cl_ds]

    elif ret_val == 'total':
        x, y, v_x, v_y, t_x, t_y = [], [], [], [], [], []
        for k in cl_ds:
            x.extend(train_clients[k][0])
            y.extend(train_clients[k][1])
            v_x.extend(val_clients[k][0])
            v_y.extend(val_clients[k][1])
            t_x.extend(test_clients[k][0])
            t_y.extend(test_clients[k][1])
        x, y = np.array(x), np.array(y)
        v_x, v_y = np.array(v_x), np.array(v_y)
        t_x, t_y = np.array(t_x), np.array(t_y)
        return x, y, v_x, v_y, t_x, t_y


def filtered_clients(num_clients, examples_range=(0, 900000), ret_val='clients'):
    train_clients, val_clients, test_clients = load_client_datasets(num_clients)
    return filter_clients(train_clients, val_clients, test_clients, num_clients, examples_range, ret_val)
