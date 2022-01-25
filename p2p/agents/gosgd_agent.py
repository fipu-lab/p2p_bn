from p2p.agents.sync_agent import *


# Distributed optimization for deep learning with gossip exchange
# Authors: Blot, Michael
#          Picard, David
#          Thome, Nicolas
#          Cord, Matthieu

# Graph topology: No graph topologies, agent communicates directly with one random agent based ond the probability p

# Code: https://github.com/uoguelph-mlrg/Theano-MPI/blob/master/theanompi/gosgd_worker.py


def choose(self_rank, high):
    """
    choose a dest_rank from range(size) to push to

    """

    dest_rank = self_rank

    while dest_rank == self_rank:
        dest_rank = np.random.randint(low=0, high=high)

    return dest_rank


def draw(p):
    """
    draw from Bernoulli distribution

    """
    # Bernoulli distribution is a special case of binomial distribution with n=1
    a_draw = np.random.binomial(n=1, p=p, size=None)

    success = (a_draw == 1)

    return success


class GoSGDAgent(SyncAgent):

    def __init__(self, p=0.1, use_graph=False, **kwargs):
        super(GoSGDAgent, self).__init__(**kwargs)

        self.w = 1.

        # p = probability of sending model after performing mini-batch update
        self.p = p

        # Our updated version can only communicate with neighbors from graph topology
        self.use_graph = use_graph
        self.make_train_iter()

    def start(self):
        # Calculate ai_w = 1 / len(nodes)
        self.w = 1. / self.graph.nodes_num
        return 0

    def train_fn(self):
        x, y = self.next_train_batch()
        self._train_on_batch(x, y)
        if draw(self.p):
            self.send_to_peers()
        return len(y)

    def send_to_peers(self):
        if self.use_graph:
            # Our modified behaviour where an agent can communicate with multiple neighbours
            peers = self.graph.get_peers(self.id)
            for peer in peers:
                self.w /= 2
                peer.receive_message(self)
        else:
            # Default behaviour, only send to one random node in the network
            self.w /= 2
            # Randomly choose a node
            a_j = choose(self.id, self.graph.nodes_num)
            other_agent = self.graph.get_node(a_j)
            other_agent.receive_message(self)

    def receive_message(self, other_agent):
        super(GoSGDAgent, self).receive_message(other_agent)
        # Formula: (ai_w * ai_m + aj_w * aj_m) / (ai_w + aj_w)
        weights = tf.nest.map_structure(lambda a, b: (a * self.w + b * other_agent.w) / (self.w + other_agent.w),
                                        self.get_model_weights(), other_agent.get_model_weights())
        self.set_model_weights(weights)
        # ai_w = ai_w + aj_w
        self.w += other_agent.w

    def sync_parameters(self):
        pass
