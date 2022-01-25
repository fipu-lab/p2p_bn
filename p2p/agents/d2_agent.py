from p2p.agents.sync_agent import *

# D2: Decentralized Training over Decentralized Data
# Authors: Tang, Hanlin
#          Lian, Xiangru
#          Yan, Ming
#          Zhang, Ce
#          Liu, Ji

# Graph topology: Ring (in experiments), Doubly stochastic (sum row and column equals 1) and symmetric/undirected
# where Wii > 0


class D2Agent(SyncAgent):
    def __init__(self, **kwargs):
        super(D2Agent, self).__init__(**kwargs)

        # Variable that holds weights from time step t-1
        self.t_1_weights = self.get_model_weights()

        self.make_train_iter()
        self.msg_q = []

    def start(self):
        # First step is just a basic train on mini-batch
        x, y = self.next_train_batch()
        return self._train_on_batch(x, y)

    def train_fn(self):
        x, y = self.next_train_batch()

        # Calculate gradients on mini-batch
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        z_grads = tape.gradient(loss, self.model.trainable_variables)

        # New model is form by formula: 2x - x(t-1)
        self.set_model_weights(tf.nest.map_structure(lambda w: w * 2, self.get_model_weights()))
        self.model.optimizer.apply_gradients(zip(z_grads, self.model.trainable_variables))

        self.set_model_weights(tf.nest.map_structure(lambda mx, mxt: mx - mxt, self.get_model_weights(), self.t_1_weights))

        self.send_to_peers()
        self.trained_examples += len(y)
        return len(y)

    def receive_message(self, other_agent):
        super(D2Agent, self).receive_message(other_agent)
        wji = self.graph.get_edge_weight(other_agent.id, self.id)
        w_xj_t2 = tf.nest.map_structure(lambda xj_t2: xj_t2 * wji, other_agent.get_model_weights())
        self.msg_q.append(w_xj_t2)

    def sync_parameters(self):
        # Multiply self model with self weight
        wii = self.graph.get_edge_weight(self.id, self.id)
        xi_tw = tf.nest.map_structure(lambda xi: xi * wii, self.get_model_weights())

        # Sum weighted models
        self.msg_q.append(xi_tw)
        self.set_model_weights(np.sum(np.array(self.msg_q, dtype=object), axis=0))
        # Remember new weights as the weights from time step t-1
        self.t_1_weights = self.get_model_weights()
        # Delete messages
        self.msg_q.clear()

