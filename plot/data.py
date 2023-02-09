import numpy as np
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
import matplotlib.pyplot as plt
import matplotlib as mpl
import string

from plot.visualize import read_json


def read_symmetric_matrix(filename):
    return symmetric_matrix(np.array(read_json(filename)))


def symmetric_matrix(X):
    i_lower = np.tril_indices(X.shape[0], -1)
    X[i_lower] = X.T[i_lower]
    return X


def seriation(Z, N, cur_index):
    """
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    """
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def create_colormesh(matrix, method=None):
    # methods = ["ward", "single", "average", "complete"]
    supported_methods = ["ward", "single", "average", "complete"]
    if method is not None and method not in supported_methods:
        raise ValueError(f"Method {method} not supported. Choose one of supported methods: {supported_methods}")

    dist_mat = squareform(pdist(matrix))
    if method is None:
        return dist_mat
    else:
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)
        return ordered_dist_mat


def plot_colormesh(matrix, method=None):
    plt.pcolormesh(create_colormesh(matrix, method))
    plt.colorbar()
    # plt.xlim([0, len(matrix)])
    # plt.ylim([0, len(matrix)])
    plt.show()


def plot_colormeshes(m_names_dict, method=None, normalize=True, fig_size=(6.4, 4.8), n_rows=1):
    fig, axs = plt.subplots(n_rows, int(len(m_names_dict) / n_rows) + (1 if len(m_names_dict) % n_rows != 0 else 0))
    axs = axs.flatten()
    if len(m_names_dict) < 2:
        axs = [axs]
    color_meshes = [create_colormesh(read_symmetric_matrix(filename), method) for filename in m_names_dict.values()]
    v_min, v_max = min([cm.min() for cm in color_meshes]), max([cm.max() for cm in color_meshes])
    if normalize:
        color_meshes = [cm / v_max for cm in color_meshes]
        v_min, v_max = 0, 1
    for ax, color_mesh, title in zip(axs, color_meshes, list(m_names_dict.keys())):
        pcm = ax.pcolormesh(color_mesh, vmin=v_min, vmax=v_max)
        ax.set_title(title)
        ax.set_xlabel(string.ascii_lowercase[ax.get_subplotspec().num1] + ")")
    for x in range(len(axs) - len(m_names_dict), 0, -1):
        fig.delaxes(axs[-1])
    # fig.colorbar(pcm, cax=fig.add_axes([0.9, 0.1, 0.03, 0.8]))
    # fig.colorbar(pcm, ax=axs, location='bottom')
    scale = 1 if fig_size is None else fig_size[1]/fig.get_figheight()
    plt.colorbar(pcm, cax=fig.add_axes([0.9, 0.088 - scale * 0.012-0.012, 0.03, 0.865 + scale * 0.09 - 0.09])) # nrow=2
    # plt.colorbar(pcm, cax=fig.add_axes([0.9, 0.06, 0.03, 0.91])) # nrow=3

    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    plt.tight_layout(pad=0, h_pad=1, w_pad=1, rect=(0, 0, 0.88, 1))
    plt.show()
    # print(fig.get_figwidth(), fig.get_figheight())


def plot_json_matrix(filename, method=None):
    plot_colormesh(read_symmetric_matrix(filename), method)


if __name__ == '__main__':
    # plot_json_matrix('data/mnist_iid_100_clients', 'complete')
    # plot_json_matrix('data/mnist_pathological-non-iid_100_clients', 'complete')
    # plot_json_matrix('data/mnist_practical-non-iid_100_clients', 'complete')
    # cm = plot_json_matrix('data/reddit_100_clients', 'complete')
    # """
    plot_colormeshes({
        'MNIST (IID)': 'data/mnist_iid_100_clients',
        'MNIST (pathological non-IID)': 'data/mnist_pathological-non-iid_100_clients',
        'MNIST (practical non-IID)': 'data/mnist_practical-non-iid_100_clients',
        'Reddit': 'data/reddit_100_clients',
        'StackOverflow': 'data/so_100_clients',
    }, 'complete', fig_size=(6.4, 4.8*1.5), n_rows=3)
    # """
