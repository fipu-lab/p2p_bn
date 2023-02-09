from tensorflow.keras.losses import KLDivergence
from p2p.agents.async_agent import *


class P2PAgent(AsyncAgent):
    # noinspection PyDefaultArgument
    def __init__(self,
                 early_stopping=True,
                 increase_momentum=False,
                 private_model_pars=None,
                 **kwargs):
        super(P2PAgent, self).__init__(**kwargs)

        self.early_stopping = early_stopping
        self.increase_momentum = increase_momentum
        self.private_model = None
        self.private_model_pars = private_model_pars
        if private_model_pars is not None:
            if private_model_pars['ds_size'] <= self.train_len:
                self.private_model = self._create_model(private_model_pars, ['model_mod', 'ds_size', 'ensemble_metrics'])

        self.kl_loss = KLDivergence()
        self.mm_decay = tf.keras.optimizers.schedules.ExponentialDecay(0.9, 5, 1.01)

        self.train_rounds = 1
        self.received_msg = False
        self.hist_ind = 1

    @property
    def _has_bn_layers(self):
        for layer in self.model.layers:
            if 'batch_normalization' in layer.name:
                return True
        return False

    def receive_message(self, other_agent):
        super(P2PAgent, self).receive_message(other_agent)
        weights = tf.nest.map_structure(lambda a, b: (a + b) / 2.0, self.get_model_weights(), other_agent.get_model_weights())
        self.set_model_weights(weights)

        self.received_msg = True
        self.train_rounds = 1

        return True

    def can_be_awaken(self):
        return self.received_msg

    def _train_on_batch(self, x, y):
        if self.has_private:
            P2PAgent._model_train_dml(self.model, self.private_model, x, y, self.kl_loss)
        else:
            Agent._model_train_batch(self.model, x, y)

    def train_epoch(self):
        if self.train_rounds < 1:
            return False

        self.model_pars['model_mod'].reset_compiled_metrics(self.model)
        if self.has_private:
            self.model_pars['model_mod'].reset_compiled_metrics(self.private_model)

        for (x, y) in self.train:
            self._train_on_batch(x, y)

        self.train_rounds = max(self.train_rounds - 1, 0)
        self.trained_examples += self.train_len

        return True

    def fit(self, epochs=0):
        if self.train_rounds < 1:
            return

        for _ in range(self.train_rounds):
            if self._has_bn_layers and self.early_stopping:
                acc_before = self.shared_val_acc()
                self.train_epoch()
                acc_after = self.shared_val_acc()
                for al1 in self.model.layers:
                    if 'batch_normalization' in al1.name:
                        if self.increase_momentum:
                            # Increasing momentum to .99 for smoother learning curve
                            al1.momentum = min(self.mm_decay(int(self.trained_examples / self.train_len)), .99)
                        continue
                    al1.trainable = acc_before < acc_after
            else:
                self.train_epoch()
        self._calc_new_metrics()

    def train_fn(self):
        self.fit()
        self.send_to_peers()
        self.received_msg = False
        return self.train_len

    def start(self):
        super(P2PAgent, self).start()
        self.fit()
        self.received_msg = True
        return self.train_len

    def shared_val_acc(self):
        for k, v in self.model_pars['model_mod'].eval_model_metrics(self.model, self.val).items():
            # TODO: Add this as a parameter, not always is acc considered
            if 'acc' in k:
                return v
        return None

    @staticmethod
    def _model_train_dml(shared_model, private_model, x, y, kl_loss):
        with tf.GradientTape(persistent=True) as tape:
            b_logits = shared_model(x, training=True)
            c_logits = private_model(x, training=True)

            bd_loss = kl_loss(b_logits, c_logits)
            cd_loss = kl_loss(c_logits, b_logits)
            b_loss = shared_model.loss(y, b_logits)
            c_loss = private_model.loss(y, c_logits)

            shared_loss = tf.add(b_loss, bd_loss)
            private_loss = tf.add(c_loss, cd_loss)

        b_grads = tape.gradient(shared_loss, shared_model.trainable_variables)
        shared_model.optimizer.apply_gradients(zip(b_grads, shared_model.trainable_variables))
        shared_model.compiled_metrics.update_state(y, b_logits)

        c_grads = tape.gradient(private_loss, private_model.trainable_variables)
        private_model.optimizer.apply_gradients(zip(c_grads, private_model.trainable_variables))
        private_model.compiled_metrics.update_state(y, c_logits)
        return tf.divide(tf.add(bd_loss, cd_loss), 2)

    def calc_new_metrics(self, metrics_names=None):
        self.hist_ind += 1
        checkpoints = len(self.hist['examples'])
        for _ in range(abs(checkpoints - self.hist_ind)):
            for _, v in self.hist.items():
                if not hasattr(v, '__iter__') or isinstance(v, str):
                    continue
                if self.hist_ind < checkpoints:
                    del v[-2]
                elif self.hist_ind > checkpoints:
                    v.append(v[-1])

    def _calc_new_metrics(self, metrics_names=None):
        super(P2PAgent, self).calc_new_metrics(metrics_names)
        if self.has_private:
            self._add_hist_metric(self._eval_train_metrics(self.private_model), "train_private", metrics_names)
            self._add_hist_metric(self._eval_val_metrics(self.private_model), "val_private", metrics_names)
            self._add_hist_metric(self._eval_test_metrics(self.private_model), "test_private", metrics_names)

            for key, dataset in zip(["train_ensemble", "val_ensemble", "test_ensemble"], [self.train, self.val, self.test]):
                self._add_hist_metric(
                    self.model_pars['model_mod'].eval_ensemble_metrics([self.model, self.private_model], dataset, self.private_model_pars['ensemble_metrics']),
                    key, metrics_names
                )

    @property
    def hist_train_private_metric(self, metric_name='acc'):
        return self._get_hist_metric("train_private", metric_name)

    @property
    def hist_train_ensemble_metric(self, metric_name='acc'):
        return self._get_hist_metric("train_ensemble", metric_name)

    @property
    def hist_val_private_metric(self, metric_name='acc'):
        return self._get_hist_metric("val_private", metric_name)

    @property
    def hist_val_ensemble_metric(self, metric_name='acc'):
        return self._get_hist_metric("val_ensemble", metric_name)

    @property
    def hist_test_private_metric(self, metric_name='acc'):
        return self._get_hist_metric("test_private", metric_name)

    @property
    def hist_test_ensemble_metric(self, metric_name='acc'):
        return self._get_hist_metric("test_ensemble", metric_name)

    @property
    def has_private(self):
        return self.private_model is not None

    @property
    def trainable(self):
        return self.train_rounds > 0

    @property
    def memory_footprint(self):
        if self.has_private:
            return self.model_pars['model_mod'].calculate_memory_model_size(self.model) + self.model_pars['model_mod'].calculate_memory_model_size(self.private_model)
        return super(P2PAgent, self).memory_footprint

    def deserialize(self):
        super(P2PAgent, self).deserialize()
        if self.has_private:
            self.private_model = self.model_pars['model_mod'].load('p2p_models/{}/agent_{}_private'.format(self.__class__.__name__, self.id))

    def serialize(self, save_only=False):
        super(P2PAgent, self).serialize(save_only)
        if self.has_private:
            self.model_pars['model_mod'].save(self.private_model, 'p2p_models/{}/agent_{}_private'.format(self.__class__.__name__, self.id))
            if not save_only:
                self.private_model = True
