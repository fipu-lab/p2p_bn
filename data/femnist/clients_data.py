from collections import defaultdict
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])

        for c_id in cdata['user_data'].keys():
            cdata['user_data'][c_id]['x'] = np.expand_dims(np.array(cdata['user_data'][c_id]['x']).astype("float32").reshape([-1, 28, 28]), -1)
            cdata['user_data'][c_id]['y'] = np.array(cdata['user_data'][c_id]['y']).astype('int8')
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def load_clients_data(num_clients=100, seed=None):
    clients, _, d_train = read_dir('data/femnist/femnist/data/train/')
    _, _, d_test = read_dir('data/femnist/femnist/data/test/')

    if seed is not None:
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        rs = RandomState(MT19937(SeedSequence(seed)))
        clients = rs.choice(clients, size=num_clients, replace=False)
    else:
        clients = np.random.choice(clients, size=num_clients, replace=False)

    c_train = [[d_train[c]['x'], d_train[c]['y']] for c in clients]
    c_test = [[d_test[c]['x'], d_test[c]['y']] for c in clients]
    c_val = []
    for (ctrx, ctry), (_, ctesty) in zip(c_train, c_test):
        image_generator = ImageDataGenerator(
            # rotation_range=10,
            # zoom_range=0.05,
            width_shift_range=0.025,
            height_shift_range=0.025,
            # horizontal_flip=False,
            # vertical_flip=False,
            # zca_whitening=False,
            data_format="channels_last")
        rnd_ind = np.random.randint(len(ctry), size=len(ctesty))
        image_generator.fit(ctrx, augment=True)
        x_augmented, y_augmented = image_generator.flow(ctrx[rnd_ind], ctry[rnd_ind], batch_size=len(rnd_ind), shuffle=False).next()[0:2]
        c_val.append([x_augmented, y_augmented])
    data = {
        "train": c_train,
        "val": c_val,
        "test": c_test,
        "dataset_name": ['femnist'] * num_clients,
    }
    return data


if __name__ == '__main__':
    train, val, test = load_clients_data()
    for t1, v1, t2 in zip(train, val, test):
        print(len(t1[0]), len(t2[0]), len(v1[0]), len(t2[0])/len(t1[0]))

