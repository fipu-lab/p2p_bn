from tensorflow import keras
import numpy as np


def shuffle_samples(x_samples, y_samples):
    assert len(x_samples) == len(y_samples)
    indices = np.arange(len(y_samples))
    np.random.shuffle(indices)
    return [x_samples[j] for j in indices], [y_samples[j] for j in indices]


def shuffle_clients_data(ds_x, ds_y):
    assert len(ds_x) == len(ds_y)
    for i in range(len(ds_y)):
        ds_x[i], ds_y[i] = shuffle_samples(ds_x[i], ds_y[i])
    return ds_x, ds_y


def split_uniform_per_label(x_data, y_data, num_splits):
    ds_x = [[] for _ in range(num_splits)]
    ds_y = [[] for _ in range(num_splits)]
    for label in np.unique(y_data):
        s = np.array_split(np.argwhere(y_data == label), num_splits)
        for i in range(num_splits):
            si = np.squeeze(s[i])
            ds_x[i].extend(x_data[si])
            ds_y[i].extend(y_data[si])

    return shuffle_clients_data(ds_x, ds_y)


def load_clients_data(num_clients=100, mode='clusters', **kwargs):
    # Mode: IID, pathological non-IID, practical non-IID
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    if mode == 'clusters':
        clusters = kwargs.get('clusters', 2)
        if isinstance(clusters, int):
            cluster_num = clusters
            clusters = []
            for c in range(cluster_num):
                cli_num = int(num_clients / cluster_num)
                clusters.append(list(range(cli_num*c, cli_num*(c+1))))

        rotations = kwargs.get('rotations', [0] * len(clusters))
        label_swaps = kwargs.get('label_swaps', [[] for _ in clusters])
        label_partitions = kwargs.get('label_partitions', [np.array(list(set(y_train))) for _ in clusters])
        samples = kwargs.get('samples', -1)
        val_ratio = kwargs.get('val_ratio', 0.2)

        # seed = kwargs.get('seed', 123)
        cluster_num = len(clusters)
        assert num_clients/cluster_num == int(num_clients/cluster_num)
        assert cluster_num == len(rotations) == len(label_swaps) == len(label_partitions)
        c_x_train, c_y_train = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]
        c_x_test, c_y_test = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]

        def do_split(x_ds, y_ds, c_x, c_y, lbl):
            ds_label = x_ds[np.squeeze(np.argwhere(y_ds == lbl))]
            cli_ind = [cl for cl, lp in zip(clusters, label_partitions) if lbl in lp]
            cli_ind = [subitem for item in cli_ind for subitem in item]
            dx_split = np.split(ds_label, len(cli_ind))
            for i, x in zip(cli_ind, dx_split):
                c_x[i].extend(np.copy(x))
                c_y[i].extend([lbl] * len(x))

        for label in set(y_train):
            do_split(x_train, y_train, c_x_train, c_y_train, label)
            do_split(x_test, y_test, c_x_test, c_y_test, label)

        def shuffle(x, y):
            ind = np.arange(len(y))
            np.random.shuffle(ind)
            return np.array(x)[ind], np.array(y)[ind]

        for i in range(num_clients):
            c_x_train[i], c_y_train[i] = shuffle(c_x_train[i], c_y_train[i])
            c_x_test[i], c_y_test[i] = shuffle(c_x_test[i], c_y_test[i])

        def swap_labels(y, sl):
            pos_0 = np.argwhere(y == sl[0])
            pos_1 = np.argwhere(y == sl[1])
            lbls_0 = y[pos_0]
            lbls_1 = y[pos_1]
            pos = np.concatenate([pos_0, pos_1])
            y[pos] = np.concatenate([lbls_1, lbls_0])

        d_names = []
        c_x_val, c_y_val = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]
        for c, (ci, rot, ls) in enumerate(zip(clusters, rotations, label_swaps)):
            d_names.extend([f'cifar10-c{c}'] * len(ci))

            for i in ci:
                c_x_train[i] = np.array([np.rot90(img, int(rot/90)) for img in c_x_train[i]])
                c_x_test[i] = np.array([np.rot90(img, int(rot/90)) for img in c_x_test[i]])
                for l_swap in ls:
                    swap_labels(c_y_train[i], l_swap)
                    swap_labels(c_y_test[i], l_swap)

                for label in set(c_y_train[i]):
                    lbl_inds = np.squeeze(np.argwhere(c_y_train[i] == label))
                    l_ind = np.random.choice(lbl_inds, int(val_ratio*len(lbl_inds)), replace=False)
                    c_x_val[i].extend(c_x_train[i][l_ind])
                    c_y_val[i].extend(c_y_train[i][l_ind])

                    tr_i = np.logical_not(np.isin(np.arange(len(c_y_train[i])), l_ind))
                    c_x_train[i] = c_x_train[i][tr_i]
                    c_y_train[i] = c_y_train[i][tr_i]

                if samples > 0:
                    if samples > len(c_y_train[i]):
                        raise ValueError(f"Samples required: {samples}, samples available: {c_y_train[i]}")
                    labels = list(set(c_y_train[i]))
                    samples_per_label = int(samples/len(labels))
                    inds = []
                    for label in labels:
                        li = np.random.choice(np.squeeze(np.argwhere(c_y_train[i] == label)),
                                              samples_per_label, replace=False)
                        inds.extend(li)
                    c_x_train[i] = c_x_train[i][inds]
                    c_y_train[i] = c_y_train[i][inds]

        c_x_train, c_y_train = shuffle_clients_data(c_x_train, c_y_train)
        c_train = list(zip(c_x_train, c_y_train))
        c_test = list(zip(c_x_test, c_y_test))
        c_val = list(zip(c_x_val, c_y_val))
        # c_val = list(zip(np.array_split(x_test, num_clients), np.array_split(y_test, num_clients)))
        data = {
            "train": c_train,
            "val": c_val,
            "test": c_test,
            "dataset_name": d_names
        }
        return data

    elif mode == 'IID':

        c_x_train, c_y_train = split_uniform_per_label(x_train, y_train, num_clients)
        c_x_test, c_y_test = split_uniform_per_label(x_test, y_test, num_clients)
        """
        c_x_train = np.array_split(x_train, num_clients)
        c_y_train = np.array_split(y_train, num_clients)
        c_x_test = np.array_split(x_test, num_clients)
        c_y_test = np.array_split(y_test, num_clients)
        """
        c_x_val, c_y_val = c_x_test, c_y_test
        d_names = ['cifar10-{}'.format(mode)] * num_clients
    elif mode == 'pathological non-IID':
        # BrendanMcMahan2017
        # Non-IID, where we first sort the data by digit label,
        # divide it into 200 shards of size 300, and assign each of 100 clients 2 shards.
        # This is a pathological non-IID partition of the data, as most clients will only have examples of two digits.
        shards, num_cls = 2, 10
        shard_num = shards * num_clients
        shuffled_ind = list(range(0, shard_num))
        # si_copy = shuffled_ind.copy()
        # np.random.shuffle(shuffled_ind)
        # shuffled_ind = np.random.permutation(shard_num)
        val_ratio = kwargs.get('val_ratio', 0.2)
        samples = kwargs.get('samples', -1)
        step = int(shard_num / num_cls)  # 20
        shuffled_ind = [shuffled_ind[ci] for cl in range(0, step) for ci in range(cl, len(shuffled_ind), step)]
        # si = 0
        # for cl in range(0, step):
        #      for ci in range(cl, len(si_copy), step):
        #         shuffled_ind[si] = si_copy[ci]
        #         si += 1
        # for ah in range(0, len(shuffled_ind)-1):
        #     if abs(shuffled_ind[ah] - shuffled_ind[ah+1]) < 20:
        #         print(shuffled_ind[ah], shuffled_ind[ah+1])

        def shard_split(x, y):
            sorted_ind = np.argsort(y)
            cls, c_count = np.unique(y, return_counts=True)
            shard_ind = []
            for c in range(0, len(cls)):
                start_ind = sum(c_count[:c])
                shard_count = int(c_count[c] / shard_num * len(cls))
                for si in range(0, int(c_count[c] / shard_count)):
                    shard_ind.append(sorted_ind[start_ind + si * shard_count: start_ind + (si + 1) * shard_count])
            shard_ind = np.array(shard_ind, dtype=object)[shuffled_ind]

            sh_inds = []
            for i in range(0, len(shard_ind), shards):
                sh_inds.append(np.concatenate([shard_ind[sh_ind] for sh_ind in range(i, i+shards)]))
            c_x, c_y = [], []
            for sh_ind in sh_inds:
                c_x.append(x[sh_ind.astype(np.int)])
                c_y.append(y[sh_ind.astype(np.int)])
            return c_x, c_y
        c_x_train, c_y_train = shard_split(x_train, y_train)
        c_x_test, c_y_test = shard_split(x_test, y_test)

        ord_ind = [g for h in range(0, int(num_cls/shards)) for g in range(h, num_clients, int(num_cls/shards))]
        c_x_train = [c_x_train[_] for _ in ord_ind]
        c_y_train = [c_y_train[_] for _ in ord_ind]
        c_x_test = [c_x_test[_] for _ in ord_ind]
        c_y_test = [c_y_test[_] for _ in ord_ind]

        c_x_val, c_y_val = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]
        for i in range(len(c_y_train)):
            for label in set(c_y_train[i]):
                lbl_inds = np.squeeze(np.argwhere(c_y_train[i] == label))
                l_ind = np.random.choice(lbl_inds, int(val_ratio*len(lbl_inds)), replace=False)
                c_x_val[i].extend(c_x_train[i][l_ind])
                c_y_val[i].extend(c_y_train[i][l_ind])

                tr_i = np.logical_not(np.isin(np.arange(len(c_y_train[i])), l_ind))
                c_x_train[i] = c_x_train[i][tr_i]
                c_y_train[i] = c_y_train[i][tr_i]

            if samples > 0:
                if samples > len(c_y_train[i]):
                    raise ValueError(f"Samples required: {samples}, samples available: {c_y_train[i]}")
                labels = list(set(c_y_train[i]))
                samples_per_label = int(samples/len(labels))
                inds = []
                for label in labels:
                    li = np.random.choice(np.squeeze(np.argwhere(c_y_train[i] == label)),
                                          samples_per_label, replace=False)
                    inds.extend(li)
                c_x_train[i] = c_x_train[i][inds]
                c_y_train[i] = c_y_train[i][inds]

        d_names = []
        for c in range(int(num_cls/shards)):
            d_names.extend([f'cifar10-c{c}'] * int(num_clients/int(num_cls/shards)))
        # for yt, ytt in zip(c_y_train, c_y_test):
        #     print(np.unique(yt, return_counts=True), np.unique(ytt))

    elif mode == 'practical non-IID':
        # Huang2020
        # A practical non-IID data
        n_groups, num_classes = 5, 10
        group_clients = int(num_clients / n_groups)
        n_group_classes = int(num_classes / n_groups)
        sample_pct = (0.8, 0.2)
        val_ratio = kwargs.get('val_ratio', 0.2)
        samples = kwargs.get('samples', -1)

        def group_split(x, y):
            groups = {_: [] for _ in range(int(num_classes/n_group_classes))}
            cls, cls_count = np.unique(y, return_counts=True)
            for i_n in range(0, num_classes, n_group_classes):
                g_ind = int(i_n / n_group_classes)
                domin_ind = []
                for i in range(i_n, i_n + n_group_classes):
                    domin_ind.append(i)
                    num_samples = int(cls_count[i] * sample_pct[0])
                    inds = np.where(y == i)[0]
                    y_samples = np.array_split(y[inds[:num_samples]], group_clients)
                    x_samples = np.array_split(x[inds[:num_samples]], group_clients)
                    y = np.delete(y, inds[:num_samples])
                    x = np.delete(x, inds[:num_samples], axis=0)
                    groups[g_ind].append((x_samples, y_samples))
                for j in np.delete(np.array(range(num_classes)), domin_ind):
                    num_samples = int(cls_count[j] * sample_pct[1] / (n_groups - 1))
                    inds = np.where(y == j)[0]
                    y_samples = np.array_split(y[inds[:num_samples]], group_clients)
                    x_samples = np.array_split(x[inds[:num_samples]], group_clients)
                    y = np.delete(y, inds[:num_samples])
                    x = np.delete(x, inds[:num_samples], axis=0)
                    groups[g_ind].append((x_samples, y_samples))

            """ Sanity check
            for gk, gv in groups.items():
                print(gk, end=' ')
                for i in range(10):
                    c = sum([len(d) for d in gv[i][1]])
                    print(round(c/cls_count[gv[i][1][0][0]], 2), '({})'.format(gv[i][1][0][0]), end=' ')
                print()
            """
            x_clients = {_: [] for _ in range(num_clients)}
            y_clients = {_: [] for _ in range(num_clients)}
            for gk, gv in groups.items():
                for cl in gv:
                    for i, ci in enumerate(list(x_clients.keys())[gk*group_clients:(gk+1)*group_clients]):
                        x_clients[ci].extend(cl[0][i])
                        y_clients[ci].extend(cl[1][i])

            return [np.array(x_clients[_]) for _ in range(num_clients)], [np.array(y_clients[_]) for _ in range(num_clients)]

        # print("Train")
        c_x_train, c_y_train = group_split(x_train, y_train)
        # print("Test")
        c_x_test, c_y_test = group_split(x_test, y_test)

        c_x_val, c_y_val = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]
        for i in range(len(c_y_train)):
            for label in set(c_y_train[i]):
                lbl_inds = np.squeeze(np.argwhere(c_y_train[i] == label))
                l_ind = np.random.choice(lbl_inds, int(val_ratio*len(lbl_inds)), replace=False)
                c_x_val[i].extend(c_x_train[i][l_ind])
                c_y_val[i].extend(c_y_train[i][l_ind])

                tr_i = np.logical_not(np.isin(np.arange(len(c_y_train[i])), l_ind))
                c_x_train[i] = c_x_train[i][tr_i]
                c_y_train[i] = c_y_train[i][tr_i]

            if samples > 0:
                if samples > len(c_y_train[i]):
                    raise ValueError(f"Samples required: {samples}, samples available: {c_y_train[i]}")
                labels = list(set(c_y_train[i]))
                samples_per_label = int(samples/len(labels))
                inds = []
                for label in labels:
                    li = np.random.choice(np.squeeze(np.argwhere(c_y_train[i] == label)),
                                          samples_per_label, replace=False)
                    inds.extend(li)
                c_x_train[i] = c_x_train[i][inds]
                c_y_train[i] = c_y_train[i][inds]

        d_names = []
        for c in range(int(num_clients/group_clients)):
            d_names.extend([f'cifar10-c{c}'] * group_clients)
        # Sanity check
        # for i in range(100):
        #     print(np.unique(c_y_train[i], return_counts=True))

        # Test data is uniformly distributed
        # c_x_test = np.array_split(x_test, num_clients)
        # c_y_test = np.array_split(y_test, num_clients)
    else:
        raise ValueError("Invalid mode")

    c_x_train, c_y_train = shuffle_clients_data(c_x_train, c_y_train)
    c_train = list(zip(c_x_train, c_y_train))
    c_test = list(zip(c_x_test, c_y_test))
    val_ratio = kwargs.get('val_ratio', 0.2)
    if val_ratio == 0.0:
        c_x_val, c_y_val = split_uniform_per_label(x_test, y_test, num_clients)
    c_val = list(zip(c_x_val, c_y_val))
    # c_val = list(zip(np.array_split(x_test, num_clients), np.array_split(y_test, num_clients)))
    # c_val = [([], []) for _ in range(len(c_train))]
    data = {
        "train": c_train,
        "val": c_val,
        "test": c_test,
        "dataset_name": d_names,
        # "dataset_name": ['cifar10-{}'.format(mode)] * num_clients,
    }
    return data
