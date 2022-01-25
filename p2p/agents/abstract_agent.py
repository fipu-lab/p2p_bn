from common.model import *
# import environ


class Agent:
    # noinspection PyDefaultArgument
    def __init__(self, train, val, test, model, graph=None, batch_size=50, eval_batch_size=50):

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train = self._create_dataset(train[0], train[1], self.batch_size)
        self.val = self._create_dataset(val[0], val[1], self.eval_batch_size)
        self.test = self._create_dataset(test[0], test[1], self.eval_batch_size)
        self.train_len = len(train[1])

        self.model = model
        self.graph = graph

        self.trained_examples = 0
        self.hist = {"examples":      [0],
                     "train_len":     self.train_len,
                     "useful_msg":    [0],
                     "useless_msg":   [0],
                     }

        self.device = None
        self.iter = None

        self.id = 0

    @staticmethod
    def _create_dataset(x, y, batch_size):
        return tf.data.Dataset.from_tensor_slices((x, y)) \
            .shuffle(batch_size) \
            .batch(batch_size) \
            .prefetch(1)

    def next_train_batch(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.make_train_iter()
            return self.next_train_batch()

    def make_train_iter(self):
        self.iter = self.train.as_numpy_iterator()

    def _get_hist_metric(self, key, metric_name='acc'):
        for k, v in self.hist.items():
            if key in k and metric_name in k:
                return v[-1]
        return None

    @property
    def hist_train_model_metric(self, metric_name='acc'):
        return self._get_hist_metric("train_model", metric_name)

    @property
    def hist_val_model_metric(self, metric_name='acc'):
        return self._get_hist_metric("val_model", metric_name)

    @property
    def hist_test_model_metric(self, metric_name='acc'):
        return self._get_hist_metric("test_model", metric_name)

    @property
    def hist_total_messages(self):
        return sum(self.hist['useful_msg'] + self.hist['useless_msg'])

    @property
    def memory_footprint(self):
        return calculate_memory_model_size(self.model)

    def deserialize(self):
        self.model = load('p2p_models/{}/agent_{}_model'.format(self.__class__.__name__, self.id))

    def serialize(self, save_only=False):
        save(self.model, 'p2p_models/{}/agent_{}_model'.format(self.__class__.__name__, self.id))
        if not save_only:
            self.model = None

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_model_weights(self):
        return self.model.get_weights()

    def _train_on_batch(self, x, y):
        Agent._model_train_batch(self.model, x, y)
        self.trained_examples += len(y)
        return len(y)

    def train_epoch(self):
        reset_compiled_metrics(self.model)

        for (x, y) in self.train:
            self._train_on_batch(x, y)

        return self.train_len

    def fit(self, epochs=1):
        for _ in range(epochs):
            self.train_epoch()

    def start(self):
        """
        placeholder for starting simulation/training
        :return:
        """
        return 0

    def train_fn(self):
        """
            Method that trains the model. Override this to provide custom logic in subclass
        :return:
        """
        return self.train_epoch()

    @staticmethod
    def _model_train_batch(model, x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = model.loss(y, logits)
        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    def send_to_peers(self):
        peers = self.graph.get_peers(self.id)
        for peer in peers:
            peer.receive_message(self)

    def receive_message(self, other_agent):
        self.hist["useful_msg"][-1] += 1
        return True

    def reject_message(self, other_agent):
        self.hist['useless_msg'][-1] += 1
        return True

    def _eval_train_metrics(self, m):
        return eval_model_metrics(m, self.train)

    def _eval_val_metrics(self, m):
        return eval_model_metrics(m, self.val)

    def _eval_test_metrics(self, m):
        return eval_model_metrics(m, self.test)

    def calc_new_metrics(self, metrics_names=None):
        self.hist["examples"].append(self.trained_examples)
        self.hist["useful_msg"].append(0)
        self.hist["useless_msg"].append(0)

        self._add_hist_metric(self._eval_train_metrics(self.model), "train_model", metrics_names)
        self._add_hist_metric(self._eval_val_metrics(self.model), "val_model", metrics_names)
        self._add_hist_metric(self._eval_test_metrics(self.model), "test_model", metrics_names)

    def _add_hist_metric(self, metrics, key, metrics_names):
        for k, v in metrics.items():
            if metrics_names is None or len(metrics_names) == 0:
                key_name = key + '-' + k
                if key_name not in self.hist:
                    self.hist[key_name] = [0]
                self.hist[key_name].append(v)
            else:
                for m_name in metrics_names:
                    if m_name in k:
                        key_name = key + '-' + k
                        if key_name not in self.hist:
                            self.hist[key_name] = [0]
                        self.hist[key_name].append(v)
