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


def create_sparse_clusters(n, num_neighbors, create_using, nodes, clusters=None, cluster_conns=1, **kwargs):
    if clusters is None:
        clusters = len(set([a.dataset_name for a in nodes]))
        print(f"Auto detected num clusters={clusters}")
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


def create_empty(n, create_using, **kwargs):
    return nx.empty_graph(n, create_using=create_using)


def create_graph_from_conn_history(m, num_neighbors, create_using):
    adj_mx = np.zeros(m.shape)
    for i in np.flip(np.argsort(np.sum(m, axis=0))):
        for ind in np.flip(np.argsort(m[:, i])):
            if sum(adj_mx[:, ind] > 0) < num_neighbors and i != ind:
                adj_mx[i, ind] = 1
            if sum(adj_mx[i, :] > 0) == num_neighbors:
                break
    return nx.from_numpy_matrix(np.asmatrix(adj_mx), create_using=create_using)


def create_aucccr_graph(n, num_neighbors, create_using, nodes, num_test_examples=100, epochs=15, np_type=np.float64, **kwargs):
    import tensorflow as tf
    from data.metrics.aucccr import recommend_clusters, build_vector

    """
    from data.metrics.aucccr import recommend_agent_clusters_centralized
    from collections import namedtuple
    Agent = namedtuple("Agent", "model test")
    agents = []

    for ni in nodes:
        tf.keras.backend.clear_session()
        ni_model = tf.keras.models.clone_model(ni.model)
        ni_model.set_weights(ni.model.get_weights())
        ni_model.compile(optimizer=ni.model.optimizer.from_config(ni.model.optimizer.get_config()),
                         loss=ni.model.loss.from_config(ni.model.loss.get_config()))
        ni_model.fit(ni.train, epochs=epochs, verbose=0)
        agents.append(Agent(ni_model, ni.test))

    def ds_len(ds):
        return sum([len(y) for x, y in ds])

    concat_ds = nodes[0].test
    ti = 1
    while ds_len(concat_ds) < num_test_examples:
        concat_ds = concat_ds.concatenate(nodes[ti].test)
        ti += 1

    threshold = 20 if len(set([a.dataset_name for a in nodes])) > 2 else 40
    clusters = recommend_agent_clusters_centralized(agents, concat_ds,
                                                    v=lambda x: np.sqrt(threshold) if x > threshold else np.sqrt(x)
                                                    )
    """

    def ds_len(ds):
        return sum([len(y) for x, y in ds])

    concat_ds = nodes[0].test
    ti = 1
    while ds_len(concat_ds) < num_test_examples:
        concat_ds = concat_ds.concatenate(nodes[ti].test)
        ti += 1

    data = []
    for ni in nodes:
        tf.keras.backend.clear_session()
        ni_model = tf.keras.models.clone_model(ni.model)
        ni_model.set_weights(ni.model.get_weights())
        ni_model.compile(optimizer=ni.model.optimizer.from_config(ni.model.optimizer.get_config()),
                         loss=ni.model.loss.from_config(ni.model.loss.get_config()))
        ni_model.fit(ni.train, epochs=epochs, verbose=0)

        v = build_vector(ni_model, concat_ds).astype(np_type)
        data.append(v)
    print('--Data vector created--')

    threshold = 20 if len(set([a.dataset_name for a in nodes])) > 2 else 40
    clusters = recommend_clusters(data, v=lambda x: np.sqrt(threshold) if x > threshold else np.sqrt(x))

    print("-------------- AUCCCR clusters --------------")
    for i, cl in enumerate(clusters):
        print(f'\t{i}. {len(cl)}', cl)

    if len(clusters) == 0:
        raise ValueError("No clusters found")

    if sum([len(c) for c in clusters]) != n:
        print("Not all agents are clustered")

    adj_mx = np.zeros((n, n))
    for cl in clusters:
        g = sparse_graph(len(cl), num_neighbors, create_using)
        m = nx.to_numpy_matrix(g)
        for ic, c in enumerate(cl):
            adj_mx[c, cl] = m[np.squeeze(np.argwhere(np.squeeze(cl) == c))]
    g = nx.from_numpy_matrix(np.asmatrix(adj_mx), create_using=create_using)
    return g


def create_d_cliques(n, nodes, k_steps=50, clique_size=10, **kwargs):
    num_labels = nodes[0].model.layers[-1].units
    agents = np.zeros((n, num_labels))
    for i, a in enumerate(nodes):
        # a_labels = np.array([y for x, y in a.train]).flatten()
        a_labels = []
        for x, y in a.train:
            a_labels.extend(y)
        a_labels = np.array(a_labels)
        values, counts = np.unique(a_labels, return_counts=True)
        agents[i, values] = counts

    dc = []
    # Random sparation into cliques
    a_ind = np.arange(len(agents))
    while len(a_ind) > 0:
        dc.append(np.random.choice(a_ind, size=clique_size, replace=False))
        a_ind = a_ind[np.logical_not(np.isin(a_ind, dc[-1]))]

    def skew(clique):
        return [abs(1/len(clique)*sum([agents[c][l] for c in clique]) - 1/len(agents)*sum([agents[c][l] for c in np.array(dc).flatten()]))
                for l in range(num_labels)]

    for k in range(k_steps):
        c_ind = np.random.choice(np.arange(len(dc)), size=2, replace=False)
        c1, c2 = dc[c_ind[0]], dc[c_ind[1]]
        s = skew(c1) + skew(c2)
        swaps = []
        for i, j in zip(c1, c2):
            s_ = skew(np.concatenate([c1[c1 != i], [j]])) + skew(np.concatenate([c2[c2 != j], [i]]))
            if s_ < s:
                swaps.append(((c1, i), (c2, j)))
        if len(swaps) > 0:
            (c1, i), (c2, j) = swaps[np.random.choice(np.arange(len(swaps)), size=1, replace=False)[0]]
            np.concatenate([c1[c1 != i], [j]])
            c1_ind = np.squeeze(np.argwhere(c1 == i))
            val = c2[np.squeeze(np.argwhere(c2 == j))]
            c2[np.squeeze(np.argwhere(c2 == j))] = c1[c1_ind]
            c1[c1_ind] = val

    # fully connected inter-clique
    adj_mx = np.zeros((n, n))
    for i, nc in enumerate(dc):
        nc_l = len(nc)
        g = _graph_type_dict['complete'](n=nc_l, create_using=nx.Graph())
        m = nx.to_numpy_array(g)
        nc = np.sort(nc)
        for nci, mi in zip(nc, m):
            adj_mx[nci, nc[mi > 0]] = 1

    for ci in range(len(dc)):
        for cj in range(ci+1, len(dc)):
            ci_ind = dc[ci][np.argsort([sum(adj_mx[i, :] > 0) for i in dc[ci]])[0]]
            cj_ind = dc[cj][np.argsort([sum(adj_mx[j, :] > 0) for j in dc[cj]])[0]]
            adj_mx[ci_ind, cj_ind] = 1
            adj_mx[cj_ind, ci_ind] = 1

    # for i in range(len(adj_mx)):
    #     print(i, "{}-{}".format(sum(adj_mx[i, :] > 0), sum(adj_mx[:, i] > 0)))

    g = nx.from_numpy_matrix(np.asmatrix(adj_mx), create_using=nx.Graph())
    return g, dc


_graph_type_dict = {
    'complete': lambda **kwargs: nx.complete_graph(kwargs['n'], create_using=kwargs['create_using']),
    'ring': lambda **kwargs: create_ring(kwargs['n'], create_using=kwargs['create_using']),
    'sparse': lambda **kwargs: sparse_graph(kwargs['n'], kwargs['num_neighbors'], create_using=kwargs['create_using']),
    'erdos_renyi': lambda **kwargs: nx.erdos_renyi_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'binomial': lambda **kwargs: nx.binomial_graph(kwargs['n'], kwargs['p'], directed=kwargs['directed']),
    'torus': create_torus,
    'grid': create_grid,
    'sparse_clusters': lambda **kwargs: create_sparse_clusters(**kwargs),
    'acc_conns': lambda **kwargs: create_empty(**kwargs),
    'aucccr': lambda **kwargs: create_aucccr_graph(**kwargs),
    'd-cliques': create_d_cliques
}


class GraphManager:

    def __init__(self, graph_type, nodes, directed=False, time_varying=-1, num_neighbors=1, **kwargs):
        self.n = len(nodes)
        self.nodes = nodes
        self.directed = directed
        self.time_varying = time_varying
        self.num_neighbors = num_neighbors
        self.graph_type = graph_type
        self.graph_data = None
        self.kwargs = kwargs
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
        if self.graph_type == 'acc_conns' and time_iter == self.nodes[0].rounds and self.nodes[0].fixed_comm:
            m = np.zeros((self.nodes_num, self.nodes_num))
            for i, n in enumerate(self.nodes):
                expected_samples = (n.top_m / self.nodes_num) * n.rounds
                peers = {k.id: v for k, v in n.selected_peers.items() if v > expected_samples}
                m[i, list(peers.keys())] = list(peers.values())
            self._nx_graph = create_graph_from_conn_history(m, self.nodes[0].n_peers,
                                                            nx.DiGraph() if self._nx_graph.is_directed() else nx.Graph())

    def _resolve_graph_type(self,  **kwargs):
        assert self.graph_type in _graph_type_dict
        graph_fn = _graph_type_dict[self.graph_type]
        params = {'n': self.n,
                  'create_using': nx.DiGraph() if self.directed else nx.Graph(),
                  'directed': self.directed,
                  'num_neighbors': self.num_neighbors,
                  'p': self.num_neighbors / self.n,
                  'nodes': self.nodes
                  }
        kwargs.update(params)
        graph = graph_fn(**kwargs)
        if isinstance(graph, list) or isinstance(graph, tuple):
            graph, self.graph_data = graph
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

    def get_peers_weights(self, node_id, include_self=False):
        nb = self._graph_neighbors(node_id)
        if not include_self:
            nb = [n for n in nb if n != node_id]
        return {n: self.get_edge_weight(node_id, n) for n in nb}

    def get_peers(self, node_id):
        nb = self._graph_neighbors(node_id)
        return [n for n in self.nodes if n.id in nb and n.id != node_id]

    def get_weighted_peers(self, node_id):
        nb = self.get_peers(node_id)
        wb = [self.get_edge_weight(node_id, n.id) for n in nb]
        return nb, wb

    def get_node(self, node_id):
        return self.nodes[node_id]

    def start(self):
        pass

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
