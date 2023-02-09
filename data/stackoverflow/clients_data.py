from data.stackoverflow.preparation import load_clients
import numpy as np


def load_clients_data(num_clients=100, seed=608361, max_train_examples=20_000):
    clients = load_clients(25_000)
    choices = [i for i, tr in enumerate(clients) if 10 < len(tr[1]) <= max_train_examples]
    if seed is not None:
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        rs = RandomState(MT19937(SeedSequence(seed)))
        clients_ids = rs.choice(choices, size=num_clients, replace=False)
    else:
        # np.random.seed(seed)
        clients_ids = np.random.choice(choices, size=num_clients, replace=False)
        # clients_ids = np.random.randint(0, len(clients), num_clients)

    train, val, test = [], [], []
    for c_id in clients_ids:
        c = clients[c_id]
        c_len = len(c[1])
        train.append([c[0][:int(c_len * 0.6)], c[1][:int(c_len * 0.6)]])
        val.append([c[0][int(c_len * 0.6):int(c_len * 0.8)], c[1][int(c_len * 0.6):int(c_len * 0.8)]])
        test.append([c[0][int(c_len * 0.8):], c[1][int(c_len * 0.8):]])
    data = {
        "train": train,
        "val": val,
        "test": test,
        "dataset_name": ['stackoverflow-nwp'] * num_clients,
    }
    return data
