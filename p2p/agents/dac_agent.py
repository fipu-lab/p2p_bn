from p2p.agents.sync_agent import *
from models.abstract_model import weights_average
import numpy as np

# Decentralized adaptive clustering of deep nets is beneficial for client collaboration
# Authors: Zec, Edvin Listo
#          Ekblom, Ebba
#          Willbo, Martin
#          Mogren, Olof
#          Girdzijauskas, Sarunas


def softmax_scale(x, tau):
    num = (x * tau).astype(np.float128)
    # 11356 => max value, higher values result in np.inf
    num[num > 11356] = 11356
    num = np.exp(num)
    # num[num == np.inf] = np.finfo(np.float128).max
    num_sum = sum(num)
    if num_sum == np.inf:
        num = (num / max(num)) * np.finfo(np.float64).max
        num_sum = sum(num)
    x_new = num / num_sum
    return x_new.astype(np.float64)


def tau_function(x, a, b):
    tau = 2 * a / (1 + np.exp(-b * x)) - a + 1
    return tau


class DacAgent(SyncAgent):
    def __init__(self, tau=30, dac_var=False, n_sampled=3, **kwargs):
        super(DacAgent, self).__init__(**kwargs)
        self.tau = tau
        self.dac_var = dac_var
        self.n_sampled = n_sampled
        self.selected_peers = None
        self.priors = None
        self.new_weights = None

    def start(self):
        self.selected_peers = np.zeros(self.graph.nodes_num)
        self.priors = np.zeros(self.graph.nodes_num)
        return super(DacAgent, self).start()

    def train_fn(self):
        if self.new_weights is not None:
            self.set_model_weights(self.new_weights)
            self.new_weights = None
        return super(DacAgent, self).train_fn()

    @property
    def priors_norm(self):
        if sum(self.priors) == 0:
            priors = np.ones(self.graph.nodes_num) / self.graph.nodes_num
            priors[self.id] = 0.0
            priors = priors / np.sum(priors)
            return priors[np.arange(len(priors)) != self.id]

        not_i_idx = np.arange(len(self.priors)) != self.id
        tau = self.tau
        if self.dac_var:
            tau = tau_function(len(self.hist['examples'])-1, self.tau, 0.2)
        return softmax_scale(self.priors[not_i_idx], tau)

    def pull_from_peers(self):
        p = np.arange(self.graph.nodes_num)
        p = p[p != self.id]
        p_norm = self.priors_norm
        indx = np.random.choice(p, min((p_norm > 0).sum(), self.n_sampled), replace=False, p=p_norm)
        peers = [p for p in self.graph.nodes if p.id in indx]
        # peers = np.random.choice(list(set(self.graph.nodes) - {self}), self.n_sampled, replace=False, p=self.priors_norm)
        for other_agent in peers:
            super(DacAgent, self).receive_message(other_agent)
            loss = self.eval_model_loss(other_agent.model, self.train)
            # Mark as selected peer and add to probability priors the reciprocal loss
            self.selected_peers[other_agent.id] += 1
            self.priors[other_agent.id] = 1 / loss

        alphas = [self.train_len] + [peer.train_len for peer in peers]
        self.new_weights = weights_average([self.get_model_weights()] + [peer.get_model_weights() for peer in peers], alphas)

        # Two-hop neighbor priors estimation
        neighbour_list = np.arange(self.graph.nodes_num)
        new_neighbours = []
        for peer_n in peers:
            new_neighbours += list(set(neighbour_list[peer_n.priors > 0])
                                   - set(neighbour_list[self.selected_peers > 0])
                                   - {self.id})
        new_neighbours = np.unique(new_neighbours)
        for j in new_neighbours:
            prior_j = []
            for k, peer_k in enumerate(peers):
                score_kj = self.graph.nodes[peer_k.id].priors[j]
                if score_kj > 0:
                    prior_j.append((self.priors[peer_k.id], score_kj))
            prior_j.sort(key=lambda x: x[0])
            self.priors[j] = prior_j[-1][1]
        
        self.hist['selected_peers'] = self.selected_peers
        self.hist['priors'] = self.priors

    def sync_parameters(self):
        self.pull_from_peers()

    def update_parameters(self):
        pass
