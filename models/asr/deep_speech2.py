import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.abstract_model import *
from models.abstract_model import eval_model_metrics as base_eval_model_metrics
from jiwer import wer, cer


def create_tokenizers():
    # The set of characters accepted in the transcription.
    characters = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ'?! "]
    # Mapping characters to integers
    char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )
    return char_to_num, num_to_char


CHAR_TO_NUM, NUM_TO_CHAR = create_tokenizers()


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim))
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1))(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            # activation="tanh",
            # recurrent_activation="sigmoid",
            return_sequences=True,
        )
        x = layers.Bidirectional(
            recurrent, merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    return model


def create_model(rnn_units=512, rnn_layers=7, input_dim=193,
                 output_dim=CHAR_TO_NUM.vocabulary_size(),
                 lr=1e-4, default_weights=True):

    model = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        rnn_units=rnn_units,
        rnn_layers=rnn_layers
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=CTCLoss,
                  metrics=[])

    if isinstance(default_weights, str):
        model.load_weights(default_weights)
    elif default_weights:
        assign_default_weights(model, 'DeepSpeech_2')

    return model


def eval_model_metrics(m, dataset):
    results = base_eval_model_metrics(m, dataset)

    if next(iter(dataset), None) is None:
        return results

    results['ctc_loss'], results["wer"], results["cer"] = calc_metrics(m, dataset)
    return results


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(NUM_TO_CHAR(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def calc_metrics(m, dataset):
    loss = []
    predictions = []
    targets = []
    for batch in dataset:
        x, y = batch
        batch_predictions = m.predict(x)
        loss.append(tf.reduce_mean(CTCLoss(y, batch_predictions)))
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            label = (
                tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode("utf-8")
            )
            targets.append(label)
    wer_score = wer(targets, predictions)
    cer_score = cer(targets, predictions)
    """
    print("-" * 100)
    print(f"Word Error Rate: {wer_score:.4f}")
    print("-" * 100)
    for i in np.random.randint(0, len(predictions), 2):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)
    """
    return np.mean(loss), wer_score, cer_score
