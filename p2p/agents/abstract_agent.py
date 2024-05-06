import tensorflow as tf


class Agent:
    # noinspection PyDefaultArgument
    def __init__(self, data, model, graph=None, data_pars=None, use_tf_function=True, eval_batch_size=50):

        self.data_pars = data_pars
        self.batch_size = data_pars['batch_size']
        self.eval_batch_size = eval_batch_size
        self.use_tf_function = use_tf_function
        train, val, test = data.pop('train'), data.pop('val'), data.pop('test')
        self.train = self._create_dataset(train[0], train[1], self.batch_size, self.data_pars.get('caching', False))
        self.val = self._create_dataset(val[0], val[1], self.eval_batch_size, True)
        self.test = self._create_dataset(test[0], test[1], self.eval_batch_size, True)

        temp_train = train[1]
        while isinstance(temp_train, tuple):
            temp_train = temp_train[0]
        self.train_len = len(temp_train)
        self._data = data

        self.model_pars = model
        self.model = self._create_model(self.model_pars, ['model_mod'])
        self.graph = graph

        self.trained_examples = 0
        self.hist = {"examples":      [0],
                     "train_len":     self.train_len,
                     "useful_msg":    [0],
                     "sent_msg":      [0],
                     "useless_msg":   [0],
                     "model_name":    self.model.name,
                     "sent_to":       [[]],
                     "received_from": [[]],
                     }
        if 'dataset_name' in self._data:
            self.hist['dataset_name'] = self._data['dataset_name']

        self.device = None
        self.iter = None

        self.id = 0

        # self._tf_train_fn = None
        if use_tf_function:
            self._tf_train_fn = tf.function(Agent._model_train_batch)
        else:
            self._tf_train_fn = Agent._model_train_batch

    @staticmethod
    def _create_model(m_pars, ignored_keys):
        return m_pars['model_mod'].create_model(**{k: v for k, v in m_pars.items() if k not in ignored_keys})

    def _create_dataset(self, x, y, batch_size, use_caching=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if hasattr(self.data_pars['agents_data'], "post_process_dataset"):
            dp = self.data_pars.copy()
            dp['batch_size'] = batch_size
            ds = self.data_pars['agents_data'].post_process_dataset(ds, dp)
        else:
            ds = ds.shuffle(batch_size).batch(batch_size)
        if use_caching:
            ds = ds.cache()
        return ds.prefetch(1)

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
    def dataset_name(self):
        return self._data.get('dataset_name', '')

    @property
    def model_name(self):
        return self.model.name

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
        return self.model_pars['model_mod'].calculate_memory_model_size(self.model)

    def deserialize(self):
        self.model = self.model_pars['model_mod'].load('p2p_models/{}/agent_{}_model'.format(self.__class__.__name__, self.id))

    def serialize(self, save_only=False):
        self.model_pars['model_mod'].save(self.model, 'p2p_models/{}/agent_{}_model'.format(self.__class__.__name__, self.id))
        if not save_only:
            self.model = None

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_model_weights(self):
        return self.model.get_weights()

    def _train_on_batch(self, x, y):
        self._tf_train_fn(self.model, x, y)
        self.trained_examples += len(y)
        return len(y)

    def train_epoch(self):
        self.model_pars['model_mod'].reset_compiled_metrics(self.model)

        for (x, y) in self.train:
            self._train_on_batch(x, y)

        return self.train_len

    def fit(self, epochs=1):
        tr_len = 0
        for _ in range(epochs):
            tr_len += self.train_epoch()
        return tr_len

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
        other_agent.hist["sent_msg"][-1] += 1
        self.hist["received_from"][-1].append(other_agent.id)
        other_agent.hist["sent_to"][-1].append(self.id)
        return True

    def reject_message(self, other_agent):
        self.hist['useless_msg'][-1] += 1
        return True

    def _eval_train_metrics(self, m):
        return self.model_pars['model_mod'].eval_model_metrics(m, self.train)

    def _eval_val_metrics(self, m):
        return self.model_pars['model_mod'].eval_model_metrics(m, self.val)

    def _eval_test_metrics(self, m):
        return self.model_pars['model_mod'].eval_model_metrics(m, self.test)

    def eval_model(self, model, dataset, metrics=None):
        res_dict = self.model_pars['model_mod'].eval_model_metrics(model, dataset)
        if metrics is not None and len(metrics) > 0:
            res_dict = {k: v for k, v in res_dict.items() if k in metrics}
        return res_dict

    def eval_model_loss(self, model, dataset):
        return self.model_pars['model_mod'].eval_model_loss(model, dataset)

    def calc_new_metrics(self, metrics_names=None):
        self.hist["examples"].append(self.trained_examples)
        self.hist["useful_msg"].append(0)
        self.hist["sent_msg"].append(0)
        self.hist["useless_msg"].append(0)
        self.hist["received_from"].append([])
        self.hist["sent_to"].append([])

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
