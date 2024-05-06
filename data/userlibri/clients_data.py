import os
import random
from typing import Tuple
import numpy as np
import tensorflow_io as tfio
import tensorflow as tf
from models.asr.deep_speech2 import CHAR_TO_NUM, NUM_TO_CHAR

root_path = 'data/userlibri/UserLibri/audio_data/test-clean/'


# @tf.function(experimental_relax_shapes=True)
def encode_sample(filepath, sr=16000, fmin=40, fmax=6000, n_fft=2048, stride=128, window=256, n_mels=256):
    audio = tf.io.read_file(filepath)
    audio = tfio.audio.decode_flac(audio, dtype=tf.int16)

    audio = audio / tf.reduce_max(audio)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    spectrogram = tfio.audio.spectrogram(audio, nfft=n_fft, window=window, stride=stride)
    spectrogram = tfio.audio.melscale(spectrogram, rate=sr, mels=n_mels, fmin=fmin, fmax=fmax)
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=80)

    return spectrogram


def parse_cli_dir(cli_dir, audio_len=2048, token_len=256, frequency_dim=193, **kwargs):
    cli_txts = [f for f in os.listdir(root_path + cli_dir) if f.endswith('.txt')]
    data_x, data_y = [], []

    for txt in cli_txts:
        audios = sorted([a for a in os.listdir(root_path + cli_dir) if txt.split('.')[0] in a and txt != a])
        with open(root_path + cli_dir + '/' + txt, "r") as t:
            desc = t.read()
        rows = desc.strip().split('\n')
        for r in rows:
            f_name = r.split(' ')[0]
            y = r.split(f_name)[-1].strip()
            if f_name + '.flac' in audios:
                full_path = os.path.join(os.getcwd(), root_path, cli_dir, f_name+'.flac')
                audio = encode_sample(full_path, n_mels=frequency_dim, **kwargs)

                label = tf.strings.upper(y)
                label = tf.strings.unicode_split(label, input_encoding="UTF-8")
                label = CHAR_TO_NUM(label)

                if audio.shape[0] > audio_len:
                    pct = (audio.shape[0]-audio_len) / audio.shape[0]
                    audio = audio[:audio_len]
                    label = label[:int((1-pct)*len(label))]
                label = label[:token_len]

                padded_audio = np.pad(audio, pad_width=((0, audio_len - audio.shape[0]), (0, 0)), constant_values=0)
                # padded_audio = tf.keras.preprocessing.image.smart_resize(np.expand_dims(audio, axis=-1), (audio_len, frequency_dim), interpolation='bilinear')
                data_x.append(padded_audio)
                data_y.append(np.pad(label, pad_width=(0, token_len-len(label)), constant_values=0))
            else:
                print(f"File {f_name} not in audios")
    return data_x, data_y


def load_clients_data(num_clients=55, test_split=0.3, **kwargs):
    assert num_clients <= 55
    data = {
        "train": [],
        "val": [([], []) for _ in range(num_clients)],
        "test": [],
        "dataset_name": ['userlibri'] * num_clients,
    }
    for ci, cli_dir in enumerate(os.listdir(root_path)[:num_clients]):
        x, y, = parse_cli_dir(cli_dir, **kwargs)
        tr_ind = random.sample(range(len(x)), k=int(len(x)*(1-test_split)))
        train = ([train_cx for xi, train_cx in enumerate(x) if xi in tr_ind],
                 [train_cy for yi, train_cy in enumerate(y) if yi in tr_ind])
        test = ([test_cx for test_xi, test_cx in enumerate(x) if test_xi not in tr_ind],
                [test_cy for test_yi, test_cy in enumerate(y) if test_yi not in tr_ind])

        data['train'].append(train)
        data['test'].append(test)

    return data


def post_process_dataset(tf_dataset, data_pars):
    if next(iter(tf_dataset), None) is None:
        return tf_dataset
    x, y = next(iter(tf_dataset))
    if data_pars.get('model', '').lower() == 'las':
        def make_example(audio: tf.Tensor, tokens: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
            lbl_ind = tf.where(tf.not_equal(tokens, tf.constant(0, dtype=tf.int32)))
            return (audio, tf.squeeze(tf.gather(tokens, lbl_ind[:-1]))), tf.squeeze(tf.gather(tokens, lbl_ind[1:]))
        tf_dataset = tf_dataset.map(make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        padded_shape = ((x.shape, y.shape[0]-1), [y.shape[0]-1])
    else:
        # DS2
        padded_shape = (x.shape, y.shape)
    tf_dataset = tf_dataset.shuffle(data_pars['batch_size']).padded_batch(data_pars['batch_size'], padded_shape)
    return tf_dataset
