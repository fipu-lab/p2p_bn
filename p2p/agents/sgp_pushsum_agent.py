from p2p.agents.sync_agent import *

# Stochastic Gradient Push using PushSum
# Stochastic Gradient-Push for Strongly Convex Functions on Time-Varying Directed Graphs
# Authors: Angelia Nedic
#          Alex Olshevsky

# Graph topology: Generic communication topologies that may be directed, sparse, time-varying
# Communication/weight matrix is column stochastic (all columns sum to 1) where Wii > 0

# See "Stochastic gradient push for distributed deep learning" for details on implementation
# In this paper, authors used directed directed exponential graph where in each iteration node only communicates with one node
# This is achieved using variable k which increases with each iteration, so choosing sending peer is peers[k%len(peers)]


class Msg:
    def __init__(self, x, w):
        self.x = x
        self.w = w


class SGPPushSumAgent(SyncAgent):
    def __init__(self, **kwargs):
        super(SGPPushSumAgent, self).__init__(**kwargs)

        self.w = 1
        # self.msg_q = []
        self.model_q = None
        self.w_q = 0
        self.make_train_iter()

    def train_fn(self):
        # Convert model to de-biased estimate
        self.set_model_weights(tf.nest.map_structure(lambda xi: xi / self.w, self.get_model_weights()))

        # Calculate gradients on de-biased model
        x, y = self.next_train_batch()
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        grad = tape.gradient(loss, self.model.trainable_variables)

        # Convert back to biased estimate
        self.set_model_weights(tf.nest.map_structure(lambda xi: xi * self.w, self.get_model_weights()))
        # Apply the newly-computed stochastic gradients to the biased model
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        self.trained_examples += len(y)
        self.send_to_peers()
        return len(y)

    def receive_message(self, other_agent):
        super(SGPPushSumAgent, self).receive_message(other_agent)
        # Weight is from the sending agent, since the communication can also be directed
        wji = self.graph.get_edge_weight(other_agent.id, self.id)

        # Append message to queue to process later
        # self.msg_q.append(Msg(x=tf.nest.map_structure(lambda o: o * wji, other_agent.get_model_weights()), w=self.w * wji))
        x = tf.nest.map_structure(lambda o: o * wji, other_agent.get_model_weights())
        w = self.w * wji
        if self.model_q is None:
            self.model_q = x
        else:
            self.model_q = tf.nest.map_structure(lambda i, j: i + j, self.model_q, x)
        self.w_q += w

    def sync_parameters(self):
        # Agent must have "connection with itself"
        wii = self.graph.get_edge_weight(self.id, self.id)
        x = tf.nest.map_structure(lambda w: w * wii, self.get_model_weights())

        # xij_s = [msg.x for msg in self.msg_q] + [x]
        self.model_q = tf.nest.map_structure(lambda i, j: i + j, self.model_q, x)
        self.set_model_weights(self.model_q)
        # self.set_model_weights(np.sum(np.array(xij_s, dtype=object), axis=0))

        # self.w = self.w * wii + sum([msg.w for msg in self.msg_q])
        self.w = self.w * wii + self.w_q
        self.w_q = 0
        self.model_q = None
        # self.msg_q.clear()
