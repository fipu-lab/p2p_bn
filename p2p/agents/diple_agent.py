from p2p.agents.sync_agent import *
from models.abstract_model import weights_average
import numpy as np

# DiPLe: Learning Directed Collaboration Graphs for Peer-to-Peer Personalized Learning
# Authors: Zheng, Xue
#          Naghizadeh, Parinaz
#          Yener, Aylin


class DiPLeAgent(SyncAgent):
    def __init__(self, t=5, epsilon=0.005, n_sampled=3, **kwargs):
        super(DiPLeAgent, self).__init__(**kwargs)
        self.t = t
        self.epsilon = epsilon
        self.n_sampled = n_sampled
        self.iteration = 0
        self.best_weights = None
        self.w = None
        self.make_train_iter()
        self.selected_peers = None

    def start(self):
        self.w = np.array([0.5] * self.graph.nodes_num)
        return super(DiPLeAgent, self).start()

    def train_fn(self):
        if self.iteration == 0:
            return super(DiPLeAgent, self).train_fn()
        else:
            self.train_batch()
            return 0

    def pull_peers(self):
        peer_ids = np.argwhere(self.w > 0).flatten()
        peer_ids = peer_ids[peer_ids != self.id]
        if self.n_sampled > 0:
            peer_ids = np.random.choice(peer_ids, size=min(len(peer_ids), self.n_sampled), replace=False)
        peers = [p for p in self.graph.nodes if p.id in peer_ids]
        for peer in peers:
            super(DiPLeAgent, self).receive_message(peer)
        self.selected_peers = peers
        return peers

    def update_communication_weights(self):
        peers = self.pull_peers()
        p_weights = self.get_model_weights()

        for peer in peers:
            peer_weights = peer.get_model_weights()

            def calc_loss(alpha):
                avg_weights = tf.nest.map_structure(lambda m1, m2: (1-alpha)*m1 + alpha*m2,
                                                    p_weights, peer_weights)
                self.set_model_weights(avg_weights)
                loss = self.eval_model_loss(self.model, self.train)
                return loss

            self.w[peer.id] = bisection_method(fn=calc_loss, a=0, b=1, epsilon=self.epsilon)
            if self.w[peer.id] < self.epsilon/2:
                self.w[peer.id] = 0

        self.set_model_weights(p_weights)
        # Normalize weights
        self.w /= self.w.sum()

    def train_batch(self):
        x, y = self.next_train_batch()
        # Calculate gradients on mini-batch
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def sync_parameters(self):
        if self.iteration == 0:
            self.update_communication_weights()
        else:
            peers = self.selected_peers
            for peer in peers:
                super(DiPLeAgent, self).receive_message(peer)

            new_weights = weights_average([p.get_model_weights() for p in [self]+peers],
                                          [self.w[p.id] for p in [self]+peers])

            self.set_model_weights(new_weights)
            loss_ = self.eval_model_loss(self.model, self.train)
            if self.best_weights is None or self.best_weights[0] > loss_:
                self.best_weights = [loss_, new_weights]

            self.train_batch()
            loss = self.eval_model_loss(self.model, self.train)
            if self.best_weights[0] > loss:
                self.best_weights = [loss, self.get_model_weights()]

    def update_parameters(self):
        self.iteration += 1
        if self.iteration > self.t:
            self.iteration = 0
            self.set_model_weights(self.best_weights[1])
            self.best_weights = None
            self.selected_peers = None

        self.hist['w'] = self.w


def bisection_method(fn, a, b, epsilon):
    while (b - a) / 2 > epsilon:
        c = (a + b) / 2
        if fn(c) == 0:
            return c
        elif fn(a) * fn(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2
