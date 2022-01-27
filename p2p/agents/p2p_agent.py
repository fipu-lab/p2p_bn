from p2p.agents.async_agent import *


class P2PAgent(AsyncAgent):
    # noinspection PyDefaultArgument
    def __init__(self,
                 early_stopping=True,
                 increase_momentum=False, 
                 **kwargs):
        super(P2PAgent, self).__init__(**kwargs)

        self.early_stopping = early_stopping
        self.increase_momentum = increase_momentum
        self.mm_decay = tf.keras.optimizers.schedules.ExponentialDecay(0.9, 5, 1.01)

        self.train_rounds = 1
        self.received_msg = False

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
        Agent._model_train_batch(self.model, x, y)

    def train_epoch(self):
        if self.train_rounds < 1:
            return False

        reset_compiled_metrics(self.model)

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
        for k, v in eval_model_metrics(self.model, self.val).items():
            if 'acc' in k:
                return v
        return None

    @property
    def trainable(self):
        return self.train_rounds > 0
