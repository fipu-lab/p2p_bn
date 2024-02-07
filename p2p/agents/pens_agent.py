from p2p.agents.sync_agent import *
from models.abstract_model import weights_average
import numpy as np

# Decentralized federated learning of deep neural networks on non-iid data
# Authors: Onoszko, Noa
#          Karlsson, Gustav
#          Mogren, Olof
#          Zec, Edvin Listo


class PensAgent(SyncAgent):
    def __init__(self, rounds=100, n_sampled=6, top_m=3, n_peers=3, fixed_comm=False, **kwargs):
        super(PensAgent, self).__init__(**kwargs)
        self.rounds = rounds
        self.n_sampled = n_sampled
        self.top_m = top_m
        self.n_peers = n_peers
        self.fixed_comm = fixed_comm
        self.iteration = 0
        self.selected_peers = {}
        self.new_weights = None

    def train_fn(self):
        if self.new_weights is not None:
            self.set_model_weights(self.new_weights)
            self.new_weights = None
        self.iteration += 1
        return super(PensAgent, self).train_fn()

    def pull_from_peers(self):
        if self.iteration < self.rounds:
            p = np.arange(self.graph.nodes_num)
            p = p[p != self.id]
            indx = np.random.choice(p, self.n_sampled, replace=False)
            peers = [p for p in self.graph.nodes if p.id in indx]
            # peers = np.random.choice(list(set(self.graph.nodes) - {self}), self.n_sampled, replace=False)
        else:
            expected_samples = (self.top_m / self.graph.nodes_num) * self.rounds
            peers = [k for k, v in self.selected_peers.items() if v > expected_samples]
            indx = np.random.choice(np.array([p.id for p in peers]), self.n_peers, replace=False)
            peers = [p for p in peers if p.id in indx]
            # peers = np.random.choice(peers, size=self.n_peers, replace=False)
            if self.fixed_comm:
                graph_peers = self.graph.get_peers(self.id)
                # If the comm matrix is built, use that peers in fixed communication
                if len(graph_peers) != 0:
                    peers = graph_peers
        for peer in peers:
            super(PensAgent, self).receive_message(peer)

        if self.iteration < self.rounds:
            saved_models = {self.eval_model_loss(p.model, self.train): p for p in peers}
            peers = list(dict(sorted(saved_models.items())).values())[:self.top_m]
            for peer in peers:
                # We want to receive more messages from this peer so mark as selected peer
                if peer not in self.selected_peers:
                    self.selected_peers[peer] = 0
                self.selected_peers[peer] += 1

        alphas = [self.train_len] + [peer.train_len for peer in peers]
        ws = [self.get_model_weights()] + [peer.get_model_weights() for peer in peers]
        self.new_weights = weights_average(ws, alphas)

        self.hist['selected_peers'] = {p.id: v for p, v in self.selected_peers.items()}

    def sync_parameters(self):
        self.pull_from_peers()

    def update_parameters(self):
        pass
