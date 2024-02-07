from p2p.agents.sync_agent import *
from models.abstract_model import weights_average


class GossipPullAgent(SyncAgent):
    def __init__(self, **kwargs):
        super(GossipPullAgent, self).__init__(**kwargs)
        self.new_weights = None

    def train_fn(self):
        if self.new_weights is not None:
            self.set_model_weights(self.new_weights)
            self.new_weights = None
        return super(GossipPullAgent, self).train_fn()

    def pull_from_peers(self):
        peers = self.graph.get_peers(self.id)
        for peer in peers:
            super(GossipPullAgent, self).receive_message(peer)

        alphas = [self.train_len] + [peer.train_len for peer in peers]
        ws = [self.get_model_weights()] + [peer.get_model_weights() for peer in peers]
        self.new_weights = weights_average(ws, alphas)

    def sync_parameters(self):
        self.pull_from_peers()

    def update_parameters(self):
        pass
