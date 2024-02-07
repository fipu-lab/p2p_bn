from p2p.agents.sync_agent import *
import numpy as np

# D-Cliques: Compensating for Data Heterogeneity with Topology in Decentralized Federated Learning
# Authors: Bellet, Aurelien
#          Kermarrec, Anne Marie
#          Lavoie, Erick


class DCliqueAgent(SyncAgent):
    def __init__(self, **kwargs):
        super(DCliqueAgent, self).__init__(**kwargs)
        self.clique = None
        self.new_weights = None
        self.make_train_iter()
        self.grads = None
        self.new_grads = None

    def start(self):
        assert self.graph.graph_type == "d-cliques", "Graph type not set to d-cliques"
        self.clique = self.graph.graph_data[np.argwhere(np.array(self.graph.graph_data) == self.id)[0][0]]
        return super(DCliqueAgent, self).start()

    def train_fn(self):
        if self.new_weights is not None:
            self.set_model_weights(self.new_weights)
            self.new_weights = None

        x, y = self.next_train_batch()

        # Calculate gradients on mini-batch
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        z_grads = tape.gradient(loss, self.model.trainable_variables)
        self.grads = z_grads

        self.trained_examples += len(y)
        return len(y)

    def sync_parameters(self):
        self.pull_gradients()

    def update_parameters(self):
        self.pull_models()
        self.hist['clique'] = self.clique

    def pull_models(self):
        peers = self.graph.get_peers(self.id)
        models = []
        for peer in peers:
            self.receive_message(peer)
            wij = self.graph.get_edge_weight(self.id, peer.id)
            models.append(tf.nest.map_structure(lambda m: m*wij, peer.get_model_weights()))

        wii = self.graph.get_edge_weight(self.id, self.id)
        weights = tf.nest.map_structure(lambda m: m*wii, self.get_model_weights())
        for model in models:
            weights = tf.nest.map_structure(lambda m1, m2: m1+m2, weights, model)

        self.new_weights = weights

    def pull_gradients(self):
        peers = self.graph.get_peers(self.id)
        clique_peers = [p for p in peers if p.id in self.clique]
        for peer in clique_peers:
            self.receive_message(peer)
            self.grads = tf.nest.map_structure(lambda g1, g2: tf.convert_to_tensor(g1)+tf.convert_to_tensor(g2),
                                               self.grads, peer.grads)
        self.grads = tf.nest.map_structure(lambda g: g / len(clique_peers), self.grads)

        self.model.optimizer.apply_gradients(zip(self.grads, self.model.trainable_variables))


