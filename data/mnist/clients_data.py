from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def load_clients_data(num_clients=100, mode='IID'):
    # Mode: IID, pathological non-IID, practical non-IID
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    if mode == 'IID':
        c_x_train = np.array_split(x_train, num_clients)
        c_y_train = np.array_split(y_train, num_clients)
        c_x_test = np.array_split(x_test, num_clients)
        c_y_test = np.array_split(y_test, num_clients)
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
                sh_inds.append(np.concatenate([shard_ind[sh_ind] for sh_ind in range(i, i + shards)]))
            c_x, c_y = [], []
            for sh_ind in sh_inds:
                c_x.append(x[sh_ind])
                c_y.append(y[sh_ind])
            return c_x, c_y

        c_x_train, c_y_train = shard_split(x_train, y_train)
        c_x_test, c_y_test = shard_split(x_test, y_test)

        # for yt, ytt in zip(c_y_train, c_y_test):
        #     print(np.unique(yt, return_counts=True), np.unique(ytt))

    elif mode == 'practical non-IID':
        # Huang2020
        # A practical non-IID data
        n_groups, num_classes = 5, 10
        group_clients = int(num_clients / n_groups)
        n_group_classes = int(num_classes / n_groups)
        sample_pct = (0.8, 0.2)

        def group_split(x, y):
            groups = {_: [] for _ in range(int(num_classes / n_group_classes))}
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
                    for i, ci in enumerate(list(x_clients.keys())[gk * group_clients:(gk + 1) * group_clients]):
                        x_clients[ci].extend(cl[0][i])
                        y_clients[ci].extend(cl[1][i])

            return [x_clients[_] for _ in range(num_clients)], [y_clients[_] for _ in range(num_clients)]

        # print("Train")
        c_x_train, c_y_train = group_split(x_train, y_train)
        # print("Test")
        c_x_test, c_y_test = group_split(x_test, y_test)

        # Sanity check
        # for i in range(100):
        #     print(np.unique(c_y_train[i], return_counts=True))

        # Test data is uniformly distributed
        # c_x_test = np.array_split(x_test, num_clients)
        # c_y_test = np.array_split(y_test, num_clients)
    else:
        raise ValueError("Invalid mode")

    c_train = list(zip(c_x_train, c_y_train))
    c_test = list(zip(c_x_test, c_y_test))
    # c_val = list(zip(np.array_split(x_test, num_clients), np.array_split(y_test, num_clients)))
    # c_val = [([], []) for _ in range(len(c_train))]
    c_val = []
    for (ctrx, ctry), (_, ctesty) in zip(c_train, c_test):
        image_generator = ImageDataGenerator(
            # rotation_range=10, # =1
            # zoom_range=0.05,
            width_shift_range=0.025,
            height_shift_range=0.025,
            data_format="channels_last")
        rnd_ind = np.random.randint(len(ctry), size=len(ctesty))
        image_generator.fit(ctrx, augment=True)
        x_augmented, y_augmented = image_generator.flow(ctrx[rnd_ind], ctry[rnd_ind], batch_size=len(rnd_ind), shuffle=False).next()[0:2]
        c_val.append([x_augmented, y_augmented])
    data = {
        "train": c_train,
        "val": c_val,
        "test": c_test,
        "dataset_name": 'mnist-{}'.format(mode),
    }
    return data
