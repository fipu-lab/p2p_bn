import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.asr.deep_speech2 import CHAR_TO_NUM, NUM_TO_CHAR


def read_data():
    data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
    wavs_path = data_path + "/wavs/"
    metadata_path = data_path + "/metadata.csv"

    # Read metadata file and parse it
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    return metadata_df, wavs_path


def encode_single_sample(wav_file_path, label, frame_length=256, frame_step=160, fft_length=384, **kwargs):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wav_file_path)  # wavs_path + wav_file + ".wav"
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.upper(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = CHAR_TO_NUM(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label


def load_clients_data(num_clients=100, test_split=0.3, train_size=-1, **kwargs):
    data = {
        "train": [],
        "val": [([], []) for _ in range(num_clients)],
        "test": [],
        "dataset_name": ['ljspeech'] * num_clients,
    }

    metadata_df, wavs_path = read_data()
    split = int(len(metadata_df) * (1-test_split))
    df_train = metadata_df[:split]
    df_test = metadata_df[split:]

    cli_train_len = int(len(df_train) / num_clients)
    cli_test_len = int(len(df_test) / num_clients)

    for i in range(num_clients):
        train = ([wavs_path+wav_file+".wav" for wav_file in list(df_train["file_name"])[i*cli_train_len:(i+1)*cli_train_len]],
                 list(df_train["normalized_transcription"])[i*cli_train_len:(i+1)*cli_train_len])
        test = ([wavs_path+wav_file+".wav" for wav_file in list(df_test["file_name"])[i*cli_test_len:(i+1)*cli_test_len]],
                list(df_test["normalized_transcription"])[i*cli_test_len:(i+1)*cli_test_len])

        if 0 < train_size < len(train[1]):
            pct = train_size / len(train[1])
            train = (train[0][:train_size], train[1][:train_size])
            test = (test[0][:int(pct*len(test[1]))], test[1][:int(pct*len(test[1]))])

        data['train'].append(train)
        data['test'].append(test)

    return data


def post_process_dataset(tf_dataset, data_pars):
    if next(iter(tf_dataset), None) is None:
        return tf_dataset
    # map_fn = lambda x: encode_single_sample(x, data_pars)
    tf_dataset = (
        tf_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(data_pars['batch_size'])
        .padded_batch(data_pars['batch_size'])
    )
    return tf_dataset
