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


def create_ring(n, create_using):
    return nx.cycle_graph(n, create_using=create_using)


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
        if num_neighbors == 2 and not create_using.is_directed():
            # create ring graph to prevent separations into groups
            return create_ring(n, create_using)
        # undirected d-regular graph (sum row == sum column)
        g = nx.random_regular_graph(num_neighbors, n)
    return g


def create_torus(**kwargs):
    assert kwargs['n'] % kwargs['num_neighbors'] == 0
    # kwargs = {'num_neighbors': 3, 'n': 15, 'create_using': nx.DiGraph()}
    if not kwargs['create_using'].is_directed():
        g = nx.grid_graph([kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors'])], [True, True])
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        return g
    else:
        """
        g = nx.grid_graph([kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors'])], [True, True])
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        mx = nx.to_numpy_matrix(g)
        for i in range(len(mx)):
            for j in range(i+1, len(mx)):
                mx[i, j] = 0

        nx.draw(g, pos=nx.spring_layout(g), with_labels=True)
        return g
        """
        raise NotImplementedError()


def create_grid(**kwargs):
    assert kwargs['n'] % kwargs['num_neighbors'] == 0
    if not kwargs['create_using'].is_directed():
        g = nx.grid_2d_graph(kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors']), [False, True], kwargs['create_using'])
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        return g
    else:
        """
        graph = nx.grid_graph([kwargs['num_neighbors'], int(kwargs['n'] / kwargs['num_neighbors'])], periodic=[False, True])
        edges = graph.edges()
        g = nx.from_edgelist(edges, nx.DiGraph())
        mx = nx.to_numpy_matrix(g)
        g = nx.from_numpy_matrix(mx, create_using=kwargs['create_using'])
        return g
        """
        raise NotImplementedError()


def create_sparse_clusters(n, num_neighbors, create_using, clusters=2, cluster_conns=1, **kwargs):
    # if 'cluster_directed' in kwargs:
    #     create_using = nx.DiGraph() if kwargs['cluster_directed'] else nx.Graph()
    # else:
    #     create_using = kwargs['create_using']

    if isinstance(clusters, list) or isinstance(clusters, tuple):
        clusters_num = len(clusters)
        cluster_inds = []
        count = 0
        for c in clusters:
            cluster_inds.append(list(range(count, count + c)))
            count += c
    else:
        clusters_num = clusters
        nc = int(n / clusters_num)
        cluster_inds = [list(range(i*nc, nc*(i+1))) for i in range(clusters_num)]

    assert cluster_conns <= min([len(x) for x in cluster_inds])

    if 'cluster_directed' in kwargs:
        cluster_directed = kwargs['cluster_directed']
    else:
        cluster_directed = create_using.is_directed()

    if int(cluster_conns) != cluster_conns and not cluster_directed:
        raise ValueError("Undirected clustered connections not yet working properly with fractions")

    adj_mx = np.zeros((n, n))
    # cluster_inds = []
    for i, nc in enumerate(cluster_inds):
        nc_l = len(nc)
        g = sparse_graph(nc_l, num_neighbors, create_using)
        m = nx.to_numpy_matrix(g)
        adj_mx[i*nc_l:nc_l*(i+1), i*nc_l:nc_l*(i+1)] = m
        # cluster_inds.append(list(range(i*nc, nc*(i+1))))

    if cluster_conns > 0:
        for i in range(len(cluster_inds)):
            for j in range(0 if cluster_directed else i+1, len(cluster_inds)):
                if i == j:
                    continue
                ci, cj = cluster_inds[i], cluster_inds[j]
                conns = int(cluster_conns * len(ci))
                ci_sorted = ci.copy()
                ci_sorted.sort(key=lambda x: sum(adj_mx[x] > 0))
                rnd_ci = np.concatenate([np.tile(ci, int(cluster_conns)),
                                         ci_sorted[:int(len(ci) * abs(cluster_conns - int(cluster_conns)))]
                                         # ci[:int(len(ci) * (cluster_conns - int(cluster_conns)))
                                         # np.random.choice(ci, size=int(len(ci) * (cluster_conns - int(cluster_conns))), replace=False)
                                         ]).astype(np.int)
                # [sum(adj_mx[:, x]) for x in cj]
                in_peers = [1/(sum(adj_mx[:, x])**100) for x in cj]
                in_peers = [x/sum(in_peers) for x in in_peers]
                if all([round(min(in_peers), 2) == round(x, 2) for x in in_peers]):
                    in_peers = None
                # in_peers
                while True:
                    rnd_cj = np.random.choice(cj, size=conns, replace=conns > len(cj), p=in_peers)
                    if len(set(zip(rnd_ci, rnd_cj))) == conns:
                        break
                adj_mx[rnd_ci, rnd_cj] = 1
                if not cluster_directed:
                    adj_mx[rnd_cj, rnd_ci] = 1

    # for i in range(n):
    #     print("{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))

    if cluster_directed and not create_using.is_directed():
        create_using = nx.DiGraph()
    g = nx.from_numpy_matrix(np.asmatrix(adj_mx), create_using=create_using)
    # nx.draw(g, node_color=[["blue", "green", "red", "yellow"][i] for i, c in enumerate(cluster_inds) for _ in c], with_labels=True)

    return g


_graph_type_dict = {
    'complete': lambda **kwargs: nx.complete_graph(kwargs['n'], create_using=kwargs['create_using']),
    'ring': lambda **kwargs: create_ring(kwargs['n'], create_using=kwargs['create_using']),
    'sparse': lambda **kwargs: sparse_graph(kwargs['n'], kwargs['num_neighbors'], create_using=kwargs['create_using']),
    'erdos_renyi': lambda **kwargs: nx.erdos_renyi_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'binomial': lambda **kwargs: nx.binomial_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'torus': create_torus,
    'grid': create_grid,
    'sparse_clusters': lambda **kwargs: create_sparse_clusters(**kwargs)
}


class GraphManager:

    def __init__(self, graph_type, nodes, directed=False, time_varying=-1, num_neighbors=1, **kwargs):
        self.n = len(nodes)
        self.nodes = nodes
        self.directed = directed
        self.time_varying = time_varying
        self.num_neighbors = num_neighbors
        self.graph_type = graph_type
        self._nx_graph = self._resolve_graph_type(**kwargs)
        self._resolve_weights_mixing()

    @property
    def nodes_num(self):
        return self.n

    def check_time_varying(self, time_iter):
        if self.time_varying > 0 and time_iter % self.time_varying == 0:
            print('Changing graph communication matrix')
            self._nx_graph = self._resolve_graph_type()
            self._resolve_weights_mixing()

    def _resolve_graph_type(self,  **kwargs):
        assert self.graph_type in _graph_type_dict
        graph_fn = _graph_type_dict[self.graph_type]
        params = {'n': self.n,
                  'create_using': nx.DiGraph() if self.directed else nx.Graph(),
                  'directed': self.directed,
                  'num_neighbors': self.num_neighbors,
                  'p': self.num_neighbors / self.n,
                  }
        kwargs.update(params)
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
        nx.draw(self._nx_graph, pos=nx.spring_layout(self._nx_graph), with_labels=True)

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
    gm = GraphManager('sparse_clusters', [DummyNode(_) for _ in range(80)], directed=True, num_neighbors=2,
                      **{'cluster_conns': 0.33, 'clusters': 4, 'cluster_directed': True})
                      # **{'cluster_directed': True, 'clusters': [10, 10, 10, 10]})
    # gm.draw()

    adj_mx = nx.to_numpy_array(gm._nx_graph)
    for no in gm.nodes:
        adj_mx[no.id, no.id] = 0
        print(no.id,#  len(gm.get_peers(no.id)),
              "{}-{}".format(sum(adj_mx[no.id, :] > 0), sum(adj_mx[:, no.id] > 0)),
              '\t', [p.id for p in gm.get_peers(no.id)])
