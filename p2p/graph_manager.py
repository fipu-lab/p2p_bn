import networkx as nx
import numpy as np


class DummyNode:
    def __init__(self, node_id):
        self.id = node_id


def sample_neighbors(client_num, num_clients, self_ind):
    c_list = list(range(client_num))
    c_list.remove(self_ind)
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]


def sparse_graph(n, num_neighbors, create_using):
    if create_using.is_directed():
        in_n = [num_neighbors] * n
        out_n = [num_neighbors] * n
        g = nx.directed_havel_hakimi_graph(in_n, out_n, create_using)
        """
        m = np.zeros(shape=(n, n))
        for i in range(n):
            nb = sample_neighbors(n, num_neighbors, i)
            m[i][nb] = 1
        g = nx.from_numpy_matrix(np.asmatrix(m), create_using=create_using)
        """
    else:
        # undirected d-regular graph (sum row == sum column)
        g = nx.random_regular_graph(num_neighbors, n)
    return g


_graph_type_dict = {
    'complete': lambda **kwargs: nx.complete_graph(kwargs['n'], create_using=kwargs['create_using']),
    'ring': lambda **kwargs: nx.cycle_graph(kwargs['n'], create_using=kwargs['create_using']),
    'sparse': lambda **kwargs: sparse_graph(kwargs['n'], kwargs['num_neighbors'], create_using=kwargs['create_using']),
    'erdos_renyi': lambda **kwargs: nx.erdos_renyi_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'binomial': lambda **kwargs: nx.binomial_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'grid': lambda **kwargs: nx.grid_2d_graph(kwargs['num_neighbors'], kwargs['n'], kwargs['periodic'],
                                              kwargs['create_using'])
}


class GraphManager:

    def __init__(self, graph_type, nodes, directed=False, time_varying=-1, num_neighbors=1):
        self.n = len(nodes)
        self.nodes = nodes
        self.directed = directed
        self.time_varying = time_varying
        self.num_neighbors = num_neighbors
        self.graph_type = graph_type
        self._nx_graph = self._resolve_graph_type()
        self._resolve_weights_mixing()

    @property
    def nodes_num(self):
        return self.n

    def check_time_varying(self, time_iter):
        if self.time_varying > 0 and time_iter % self.time_varying == 0:
            print('Changing graph communication matrix')
            self._nx_graph = self._resolve_graph_type()
            self._resolve_weights_mixing()

    def _resolve_graph_type(self):
        assert self.graph_type in _graph_type_dict
        graph_fn = _graph_type_dict[self.graph_type]
        kwargs = {'n': self.n,
                  'create_using': nx.DiGraph() if self.directed else nx.Graph(),
                  'directed': self.directed,
                  'num_neighbors': self.num_neighbors
                  }
        graph = graph_fn(**kwargs)
        return graph

    def _resolve_weights_mixing(self):
        for node in self._nx_graph.nodes:
            # For now, all algorithms require Wii > 0, so we add self as an edge
            self._nx_graph.add_edge(node, node)
            nbs = self._graph_neighbors(node)
            # For now, uniform weights
            weight = 1 / len(nbs)
            for nb in nbs:
                self._nx_graph[node][nb]['weight'] = weight

    def get_edge_weight(self, i, j):
        edge_data = self._nx_graph.get_edge_data(i, j)
        if edge_data is None:
            return 0
        return edge_data['weight']

    def get_self_node_weight(self, node_id):
        return self.get_edge_weight(node_id, node_id)

    def _graph_neighbors(self, node_id):
        return list(self._nx_graph.neighbors(node_id))

    def get_peers(self, node_id):
        nb = self._graph_neighbors(node_id)
        return [n for n in self.nodes if n.id in nb and n.id != node_id]

    def get_weighted_peers(self, node_id):
        nb = self.get_peers(node_id)
        wb = [self.get_edge_weight(node_id, n.id) for n in nb]
        return nb, wb

    def get_node(self, node_id):
        return self.nodes[node_id]

    def draw(self):
        nx.draw(self._nx_graph)

    def graph_info(self):
        nb_num = self.num_neighbors
        if self.graph_type == 'ring':
            nb_num = 1 if self.directed else 2
        elif self.graph_type == 'complete':
            nb_num = self.n - 1
        info = "{} ({}), N: {}, NB: {}, TV: {}".format(self.graph_type,
                                                       'directed' if self.directed else 'undirected',
                                                       self.n, nb_num,
                                                       self.time_varying)
        return info

    def as_numpy_array(self):
        return nx.to_numpy_array(self._nx_graph)


def nx_graph_from_saved_lists(np_array, directed=False):
    return nx.from_numpy_array(np.asarray(np_array), create_using=nx.DiGraph() if directed else nx.Graph())


if __name__ == "__main__":
    gm = GraphManager('sparse', [DummyNode(_) for _ in range(10)], directed=False, num_neighbors=3)
    gm.draw()
