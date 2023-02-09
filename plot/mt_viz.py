from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline
from scipy.stats import ttest_ind
import numpy as np


def plot_sparse_v_clustered(fig_size=(10, 5), nodes_num=10, node_size=300):
    import networkx as nx
    fig, axs = plt.subplots(1, 3)
    axs = axs.flatten()
    from p2p.graph_manager import GraphManager, DummyNode
    sparse = GraphManager('sparse', [DummyNode(_) for _ in range(nodes_num)], directed=True, num_neighbors=3)
    clustered = GraphManager('sparse_clusters', [DummyNode(_) for _ in range(nodes_num)], directed=True, num_neighbors=2,
                             **{'cluster_conns': 1, 'cluster_directed': True})

    task_spec = GraphManager('sparse_clusters', [DummyNode(_) for _ in range(nodes_num)], directed=True, num_neighbors=3,
                             **{'cluster_conns': 0, 'cluster_directed': True})

    def remove_self(nx_graph):
        for i in range(nx_graph.number_of_nodes()):
            if nx_graph.has_edge(i, i):
                nx_graph.remove_edge(i, i)
    remove_self(sparse._nx_graph)
    remove_self(clustered._nx_graph)
    remove_self(task_spec._nx_graph)
    colors = ["blue"] * int(nodes_num/2) + ["red"] * int(nodes_num/2)
    nx.draw(sparse._nx_graph, node_size=node_size, ax=axs[0], node_color=colors)
    nx.draw(task_spec._nx_graph, node_size=node_size, ax=axs[1], node_color=colors)
    nx.draw(clustered._nx_graph, node_size=node_size, ax=axs[2], node_color=colors)
    axs[0].set_title('Sparse', fontsize=18)
    axs[1].set_title('Task-specific clusters', fontsize=18)
    axs[2].set_title('Clustered sparse', fontsize=18)

    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])
    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
    plt.show()


def plot_4_clustered(fig_size=(10, 5), nodes_num=12, node_size=300):
    from p2p.graph_manager import GraphManager, DummyNode
    import networkx as nx
    fig, axs = plt.subplots(1, 1)
    # axs = axs.flatten()

    nodes_num=12
    graph = GraphManager('sparse_clusters', [DummyNode(_) for _ in range(nodes_num)], directed=True, num_neighbors=2,
                         **{'cluster_conns': 0.34, 'clusters': 4, 'cluster_directed': True})

    def remove_self(nx_graph):
        for i in range(nx_graph.number_of_nodes()):
            if nx_graph.has_edge(i, i):
                nx_graph.remove_edge(i, i)
    remove_self(graph._nx_graph)
    adj_mx = nx.to_numpy_array(graph._nx_graph)
    for no in graph.nodes:
        adj_mx[no.id, no.id] = 0
        print(no.id,#  len(gm.get_peers(no.id)),
              "{}-{}".format(sum(adj_mx[no.id, :] > 0), sum(adj_mx[:, no.id] > 0)),
              '\t', [p.id for p in graph.get_peers(no.id)])

    colors = ["blue"] * int(nodes_num/4) + ["red"] * int(nodes_num/4) + ["green"] * int(nodes_num/4) + ["yellow"] * int(nodes_num/4)
    nx.draw(graph._nx_graph, node_size=node_size, ax=axs, node_color=colors)
    # axs.set_title('Clustered sparse with 4 agent clusters', fontsize=18)

    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])
    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
    plt.show()


def add_titles(fig):
    fig.subplots_adjust(hspace=0.6, top=0.9)
    font = {'color': 'black', 'weight': 'heavy', 'size': 20}
    box_style = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # titles = ("Multi-task without BERT freezing", "Multi-task with BERT freezing")
    pad = " " * 100 # 83
    titles = (pad + "  MT  " + pad, pad + "MT-BF" + pad)
    fig.text(x=0.5, y=0.95, s=titles[0], fontdict=font, bbox=box_style, ha='center')
    fig.text(x=0.5, y=0.44, s=titles[1], fontdict=font, bbox=box_style, ha='center')


def exp_1():
    colors = ['r', 'g', 'b', 'indigo']  # , 'orange']
    viz = {
        'Reddit (NWP)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Reddit+StackOverflow': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_01-01-2023_18_17',
                                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_09_51',
                                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_04_03'],

                'Reddit+CoNNL': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_31-12-2022_22_57',
                                 'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_19_22',
                                 'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_22_37'],

                'Reddit+Few-NERD': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_01-01-2023_06_26',
                                    'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_22_32',
                                    'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_20_47'],
                # 'Reddit+StackOverflow (mono)': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_04_48'],
            },
        },
        'StackOverflow (NWP)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'StackOverflow+Reddit': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_01-01-2023_18_17',
                                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_09_51',
                                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_04_03'],

                'StackOverflow+CoNNL': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_31-12-2022_23_54',
                                        'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_12_11',
                                        'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_11_06'],

                'StackOverflow+Few-NERD': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_01-01-2023_20_04',
                                           'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_08_32',
                                           'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_08_04'],
                # 'StackOverflow+Reddit (mono)': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_04_48'],
            },
        },
        'CoNNL (NER)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'conll-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'CoNLL': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27'],

                'CoNNL+Reddit': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_31-12-2022_22_57',
                                 'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_19_22',
                                 'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_22_37'],

                'CoNNL+StackOverflow': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_31-12-2022_23_54',
                                        'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_12_11',
                                        'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_11_06'],

                'CoNNL+Few-NERD': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_06_59',
                                   'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_23_20',
                                   'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_23_18'],
            },
        },
        'Few-NERD (NER)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'few_nerd-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'Few-NERD': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28'],

                'Few-NERD+Reddit': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_01-01-2023_06_26',
                                    'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_22_32',
                                    'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_13-01-2023_20_47'],

                'Few-NERD+StackOverflow': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_01-01-2023_20_04',
                                           'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_08_32',
                                           'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_08_04'],

                'Few-NERD+CoNNL': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_06_59',
                                   'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_23_20',
                                   'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_14-01-2023_23_18'],
            },
        }
    }
    """
    side_by_side(viz, axis_lim=[
        {'y': [5, 12], 'step': 1},
        {'y': [5, 12], 'step': 1},
        {'y': [40, 58], 'step': 2},
        {'y': [20, 34], 'step': 2},
    ], n_rows=2, fig_size=(9, 9))
    # """

    # MT
    viz['Reddit (NWP) '] = {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Reddit+StackOverflow': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_18_20',
                                         'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_07_01',
                                         'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_06_25'],

                'Reddit+CoNNL': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_01_18',
                                 'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_10-01-2023_16_34',
                                 'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_06_27'],

                'Reddit+Few-NERD': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_17_13',
                                    'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_11_07',
                                    'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_18_11'],
            }
    }
    viz['StackOverflow (NWP) '] = {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'StackOverflow+Reddit': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_18_20',
                                         'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_07_01',
                                         'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_06_25'],

                'StackOverflow+CoNNL': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_01_17',
                                        'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_17_48',
                                        'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_19_46'],

                'StackOverflow+Few-NERD': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_04-01-2023_19_09',
                                           'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_02_35',
                                           'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_23_24'],
            }
    }
    viz['CoNNL (NER) '] = {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'conll-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'CoNLL': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27'],

                'CoNNL+Reddit': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_01_18',
                                 'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_10-01-2023_16_34',
                                 'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_06_27'],

                'CoNNL+StackOverflow': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_01_17',
                                        'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_17_48',
                                        'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_11-01-2023_19_46'],

                'CoNNL+Few-NERD': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_11_58',
                                   'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_10-01-2023_02_10',
                                   'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_10-01-2023_18_37'],
            }
    }
    viz['Few-NERD (NER) '] = {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'few_nerd-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'Few-NERD': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28'],

                'Few-NERD+Reddit': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_17_13',
                                    'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_11_07',
                                    'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_18_11'],

                'Few-NERD+StackOverflow': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_04-01-2023_19_09',
                                           'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_02_35',
                                           'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_09-01-2023_23_24'],

                'Few-NERD+CoNNL': ['mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_03-01-2023_11_58',
                                   'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_10-01-2023_02_10',
                                   'mt/sparse/mt/BertAgent_40A_300E_50B_sparse(directed-3)_10-01-2023_18_37'],
            }
    }
    # """
    fig, axs = side_by_side(viz, axis_lim=[
        {'y': [4, 11], 'step': 1},
        {'y': [4, 11], 'step': 1},
        {'y': [40, 57], 'step': 2},
        {'y': [20, 33], 'step': 2},
        {'y': [4, 11], 'step': 1},
        {'y': [4, 11], 'step': 1},
        {'y': [40, 57], 'step': 2},
        {'y': [20, 33], 'step': 2},
    ], n_rows=2, fig_size=(9*2, 9))
    add_titles(fig)
    # """
    viz['Reddit+StackOverflow (mono)'] = {
            'x_axis': 'epoch',
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Mono': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_04_48',
                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_09_50',
                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_21_44']
            }
    }
    viz['StackOverflow+Reddit (mono)'] = {
            'x_axis': 'epoch',
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'Mono': ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_02-01-2023_04_48',
                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_09_50',
                         'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_21_44']
            }
    }
    # """
    print("Statistic")
    for vk, vv in viz.items():
        print(vk)
        baseline = None
        baseline_max_a = None
        for k, v in vv['viz'].items():
            t, accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1:]
            max_a = round(max(t), 2)
            # print('\t', k, "{:.2f}".format(max_a) + '\\%', end=' ')
            # max_accs_points = [np.max(a) for a in accs]
            if baseline is None:
                baseline_max_a = max_a
                # baseline = max_accs_points
                baseline = t[40:]
                print('\t', k, "{:.2f}".format(max_a) + '\\%')
            else:
                rel_inc = round((max_a - baseline_max_a) / baseline_max_a * 100, 2)
                p_val = ttest_ind(baseline, t[40:])[1]
                p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                if rel_inc > 0:
                    p_text = "\\textbf{" + p_text + "}"
                if p_val < 0.05:
                    p_text += " \\textbf{**}"
                    # p_text = "\\emph{" + p_text + "}"
                print('\t', k, p_text)
                # print('\t\t\tp-value:', ttest_ind(baseline, t[40:])[1])
    # """


def exp_2():
    colors = ['r', 'g', 'b', 'indigo']  # , 'orange']
    viz = {
        'Reddit (NWP)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Reddit+StackOverflow': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_44',
                                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_06_39',
                                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_19_11'],

                'Reddit+CoNNL': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_18_54',
                                 'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_03_21',
                                 'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_01_29'],

                'Reddit+Few-NERD': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_07_57',
                                    'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_08_00',
                                    'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_06_07'],
                # 'Reddit+StackOverflow (mono)': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_03-01-2023_16_21'],
            }
        },
        'StackOverflow (NWP)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'StackOverflow+Reddit': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_44',
                                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_06_39',
                                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_19_11'],

                'StackOverflow+CoNNL': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_14_33',
                                        'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_11_40',
                                        'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_10_57'],

                'StackOverflow+Few-NERD': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_21_07',
                                           'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_05_52',
                                           'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_01_55'],

                # 'StackOverflow+Reddit (mono)': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_03-01-2023_16_21'],
            }
        },
        'CoNNL (NER)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'conll-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'CoNLL': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27'],

                'CoNNL+Reddit': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_18_54',
                                 'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_03_21',
                                 'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_01_29'],

                'CoNNL+StackOverflow': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_14_33',
                                        'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_11_40',
                                        'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_10_57'],

                'CoNNL+Few-NERD': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_07_39',
                                   'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_07_20',
                                   'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_09_06'],
            }
        },
        'Few-NERD (NER)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'few_nerd-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'Few-NERD': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28'],

                'Few-NERD+Reddit': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_07_57',
                                    'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_08_00',
                                    'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_06_07'],

                'Few-NERD+StackOverflow': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_21_07',
                                           'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_05_52',
                                           'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_01_55'],

                'Few-NERD+CoNNL': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_07_39',
                                   'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_07_20',
                                   'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_09_06'],
            }
        },
        #  ---- MT ----
        'Reddit (NWP) ': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Reddit+StackOverflow': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_39',
                                         'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_51',
                                         'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_07_00'],

                'Reddit+CoNNL': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_18_28',
                                 'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_17-01-2023_01_59',
                                 'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_39'],

                'Reddit+Few-NERD': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_22_59',
                                    'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_15_21',
                                    'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_18_02'],
            },
        },
        'StackOverflow (NWP) ': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'StackOverflow+Reddit': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_39',
                                         'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_51',
                                         'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_07_00'],

                'StackOverflow+CoNNL': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_11_16',
                                        'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_23_25',
                                        'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_22_25'],

                'StackOverflow+Few-NERD': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_03_31',
                                           'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_20_10',
                                           'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_19_23'],
            },
        },
        'CoNNL (NER) ': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'conll-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'CoNLL': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27'],

                'CoNNL+Reddit': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_18_28',
                                 'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_17-01-2023_01_59',
                                 'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_39'],

                'CoNNL+StackOverflow': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_11_16',
                                        'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_23_25',
                                        'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_22_25'],

                'CoNNL+Few-NERD': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_13_18',
                                   'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_03',
                                   'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_16'],
            },
        },
        'Few-NERD (NER) ': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'few_nerd-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'Few-NERD': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28'],

                'Few-NERD+Reddit': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_22_59',
                                    'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_15_21',
                                    'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_18_02'],

                'Few-NERD+StackOverflow': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_03_31',
                                           'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_20_10',
                                           'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_19_23'],

                'Few-NERD+CoNNL': ['mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_13_18',
                                   'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_03',
                                   'mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_16'],

            },
        }
    }

    fig, axs = side_by_side(viz, axis_lim=[
            {'y': [4, 12], 'step': 1},
            {'y': [4, 13], 'step': 1},
            {'y': [40, 57], 'step': 2},
            {'y': [20, 34], 'step': 2},
            {'y': [4, 12], 'step': 1},
            {'y': [4, 13], 'step': 1},
            {'y': [40, 57], 'step': 2},
            {'y': [20, 34], 'step': 2},
        ], n_rows=2, fig_size=(9*2, 9))
    add_titles(fig)
    viz['Reddit+StackOverflow (mono)'] = {
            'x_axis': 'epoch',
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Mono': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_03-01-2023_16_21',
                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_18_00',
                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_03_17'],
            }
    }
    viz['StackOverflow+Reddit (mono)'] = {
            'x_axis': 'epoch',
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'Mono': ['mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_03-01-2023_16_21',
                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_18_00',
                         'mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_03_17'],
            }
    }
    # """
    print("Statistic")
    for vk, vv in viz.items():
        print(vk)
        baseline = None
        baseline_max_a = None
        for k, v in vv['viz'].items():
            t, accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1:]
            max_a = round(max(t), 2)
            # print('\t', k, "{:.2f}".format(max_a) + '\\%', end=' ')
            # max_accs_points = [np.max(a) for a in accs]
            if baseline is None:
                baseline_max_a = max_a
                # baseline = max_accs_points
                baseline = t[40:]
                print('\t', k, "{:.2f}".format(max_a) + '\\%')
            else:
                # print('p-value:', ttest_ind(baseline, max_accs_points)[:])
                rel_inc = round((max_a - baseline_max_a) / baseline_max_a * 100, 2)
                p_val = ttest_ind(baseline, t[40:])[1]
                """
                if p_val < 10e-5:
                    p_val = "<10^{-5}"
                elif p_val < 10e-3:
                    p_val = "<10^{-3}"
                elif p_val < 5 * 10e-3:
                    p_val = "<5\\times 10^{-3}"
                else:
                    p_val = ">" + str(p_val)[:4]
                """
                p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                if rel_inc > 0:
                    p_text = "\\textbf{" + p_text + "}"
                if p_val < 0.05:
                    p_text += " \\textbf{**}"
                # if p_val > 0.05:
                    # p_text = "\\emph{" + p_text + "}"
                print('\t', k, p_text)
                # print('({}\\%, $p{}$)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc), p_val))
                # print('\t\tp-value:', ttest_ind(baseline, t[40:])[:])
    # """


def exp_4x():
    colors = ['r', 'g', 'b', 'indigo']  # , 'orange']
    viz = {
        'Reddit (NWP)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'reddit-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Reddit': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31',
                           'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01'],

                'Reddit+StackOverflow+CoNNL+Few-NERD': ['mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_08-01-2023_05_46',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_09-01-2023_17_52',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_11-01-2023_03_45']
            },
        },
        'StackOverflow (NWP)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy',
            'viz': {
                'Stackoverflow': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13',
                                  'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11'],

                'StackOverflow+Reddit+CoNNL+Few-NERD': ['mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_08-01-2023_05_46',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_09-01-2023_17_52',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_11-01-2023_03_45']
            },
        },
        'CoNNL (NER)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'conll-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'CoNLL': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07',
                          'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27'],

                'CoNNL+Reddit+StackOverflow+Few-NERD': ['mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_08-01-2023_05_46',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_09-01-2023_17_52',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_11-01-2023_03_45']
            },
        },
        'Few-NERD (NER)': {
            'x_axis': 'epoch',
            'colors': colors,
            'metric': 'few_nerd-bert-ner->test_model-macro_avg_f1_score',
            'viz': {
                'Few-NERD': ['mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32',
                             'mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28'],

                'Few-NERD+Reddit+StackOverflow+CoNNL': ['mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_08-01-2023_05_46',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_09-01-2023_17_52',
                                                        'mt/4x/BertAgent_80A_300E_50B_sparse_clusters(directed-2)_11-01-2023_03_45']
            },
        }
    }
    side_by_side(viz, axis_lim=[
        {'y': [5, 13], 'step': 1},
        {'y': [5, 13], 'step': 1},
        {'y': [40, 58], 'step': 2},
        {'y': [20, 36], 'step': 2},
    ], n_rows=2, fig_size=(9, 9))

    print("Statistic")
    for vk, vv in viz.items():
        print(vk)
        baseline = None
        baseline_max_a = None
        for k, v in vv['viz'].items():
            t, accs = parse_timeline(None, v, x_axis='examples', metric=vv['metric'])[1:]
            max_a = round(max(t), 2)
            # print('\t', k, "{:.2f}".format(max_a) + '\\%', end=' ')
            # max_accs_points = [np.max(a) for a in accs]
            if baseline is None:
                baseline_max_a = max_a
                # baseline = max_accs_points
                baseline = t[40:]
                print('\t', k, "{:.2f}".format(max_a) + '\\%')
            else:
                # print('p-value:', ttest_ind(baseline, max_accs_points)[:])
                rel_inc = round((max_a - baseline_max_a) / baseline_max_a * 100, 2)
                p_val = ttest_ind(baseline, t[40:])[1]
                """
                if p_val < 10e-5:
                    p_val = "<10^{-5}"
                elif p_val < 10e-3:
                    p_val = "<10^{-3}"
                elif p_val < 5 * 10e-3:
                    p_val = "<5\\times 10^{-3}"
                else:
                    p_val = ">" + str(p_val)[:4]
                """
                p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                if rel_inc > 0:
                    p_text = "\\textbf{" + p_text + "}"
                if p_val < 0.05:
                    p_text += " \\textbf{**}"
                # if p_val > 0.05:
                    # p_text = "\\emph{" + p_text + "}"
                print('\t', k, p_text)
                # print('({}\\%, $p{}$)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc), p_val))
                # print('\t\tp-value:', ttest_ind(baseline, t[40:])[:])
    # """


def fixer():
    # f = ['mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_19_22']
    # parse_timeline(None, f, metric='reddit-bert-nwp->test_model-sparse_categorical_accuracy')
    from plot.visualize import read_json
    f_path = 'mt/sparse/avg/BertAgent_40A_300E_50B_sparse(directed-3)_12-01-2023_22_37'
    data = read_json(f_path)
    for i in range(40):
        for k, v in data['agents'][str(i)].items():
            if isinstance(v, list):
                if len(v) > 301:
                    print(k, len(v))
                    data['agents'][str(i)][k] = v[:301]
    from p2p.p2p_utils import save_json
    save_json('log/' + f_path + '.json', data)


if __name__ == '__main__':
    # exp_1()
    # exp_2()
    exp_4x()
    # plot_4_clustered()
    # plot_sparse_v_clustered()
    # import  numpy as np
    # np.mean([3.96, -2.64, 1.1, 9.76, 0.89, 2.22, 0.51, 0.56, 6.42, -4.58, -2.24, -0.25])
    # np.mean([])

