from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline, read_graph, read_agents
from scipy.stats import ttest_ind
from data.metrics.gini import gini
import numpy as np


def print_table_2(viz, start_epoch=20, metric='avg'):
    ### Model Conns Lol Politics
    ### Gru
    ###       Solo  0   0
    print("----Table----")
    print("\\begin{table*}[b]\n\\centering\n\\footnotesize\n\\begin{tabular}{l l c c}\n\\hline")
    g1, g2 = np.unique([k[k.index('(') + 1:-1] for k in viz.keys()])
    print(
        "\\textbf{Model (dataset)} & \\textbf{Connections} &  \\textbf{leagueoflegends} &  \\textbf{politics}  \\\\\n\\hline")
    baselines = {}
    for model in ['GRU', 'BERT']:
        for group in ['small', 'big']:
            print(model + f" ({group})" + "  & & & \\\\\n\\hline")
            for col in ['Solo', 'Sparse', 'Sparse-clustered', 'Acc', 'Acc (val)', 'AUCCCR']:
                print(f"& {col} ", end='')
                for g in [g1, g2]:
                    k = f"{model} - {group} ({g})"
                    v = viz[k]
                    vv = v['viz'][col]
                    if metric == 'avg':
                        t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                        max_a = round(max(t), 2)
                    else:
                        agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                        avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                        max_a = round(float(avg_agents_max), 2)
                        t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)
                        start_epoch = 0

                    # b_key = col + "-" + model + ' (' + group + ')'
                    b_key = f"{g} -> {model} ({group})"
                    if b_key not in baselines:
                        print("&", max_a, " " if g == g1 else '\\\\', end='')
                        if col == 'Solo':  # 'Sparse':
                            baselines[b_key] = [t[start_epoch:], max_a]
                    else:
                        baseline = baselines[b_key]
                        rel_inc = round((max_a - baseline[1]) / baseline[1] * 100, 2)
                        p_val = ttest_ind(baseline[0], t[start_epoch:])[1]
                        p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format(
                            '+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                        if rel_inc > 0:
                            p_text = "\\textbf{" + p_text + "}"
                        if p_val < 0.05:
                            p_text += " \\textbf{**}"
                        print("&", p_text, " " if g == g1 else '\\\\', end='')
                print()
            print('\\hline')

    print("\\end{tabular}\n\\caption{\label{tbl:name} " + metric + ".}\n\\end{table*}")


def neigh():
    from matplotlib.colors import ListedColormap
    from p2p.graph_manager import GraphManager, DummyNode

    def plot_graph(gm=None, mx=None):
        if mx is None:
            mx = gm.as_numpy_array()
        # remove self connections
        np.fill_diagonal(mx, 0)
        mx[mx > 0] = 1
        for i in range(int(mx.shape[0] / 2)):
            mx[i][mx[i] > 0] = 3
            mx[i, int(mx.shape[0] / 2):][(mx[i] > 0)[int(mx.shape[0] / 2):]] = 2
        for i in range(int(mx.shape[0] / 2), mx.shape[0]):
            mx[i, :int(mx.shape[0] / 2)][(mx[i] > 0)[:int(mx.shape[0] / 2)]] = 2
        plt.pcolormesh(mx, cmap=ListedColormap(['white', 'blue', 'green', 'red']))
        plt.plot([50, 50], [0, 100], [0, 100], [50, 50], color='lightgray')
        plt.xlabel('Agent ID')
        plt.ylabel('Agent ID')

    plot_graph(GraphManager('sparse_clusters', [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=3,
                            cluster_conns=0))
    plot_graph(GraphManager('sparse', [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=3))
    plot_graph(GraphManager('sparse_clusters', [DummyNode(_) for _ in range(100)], directed=True, num_neighbors=2,
                            cluster_conns=1))
    # SMALL
    plot_graph(mx=np.array(
        read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_16_10')))  # AUCCCR
    plot_graph(mx=np.array(
        read_graph('conns/exp1/small/P2PAgent_100A_100E_50B_aucccr(directed-3)_19-02-2023_20_34')))  # AUCCCR clusters
    plot_graph(mx=np.array(read_graph(
        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_27-02-2023_08_46')))  # Accuracy connections
    plot_graph(mx=np.array(read_graph(
        'conns/exp1/small/P2PAgent_100A_100E_50B_acc_conns(directed-3)_10-03-2023_00_28')))  # Accuracy val connections


def plot_communication():
    def plot_graph(path):
        agents = read_agents(path)
        mx = np.zeros((len(agents), len(agents)))
        for ak, av in agents.items():
            # Flatten list
            sent = [subitem for item in av['sent_to'] for subitem in item]
            values, counts = np.unique(sent, return_counts=True)
            mx[int(ak), values] = counts
        plt.imshow(mx, cmap="GnBu")
        plt.colorbar()
        plt.show()

    # 2 Clusters #
    # Oracle
    # plot_graph()
    # Sparse
    # plot_graph()
    # Sparse clusters
    # plot_graph()
    # AUCCCR
    # plot_graph()
    # DAC


ROT_2C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_27',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_21_(1)',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_22_(1)',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_03',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_11_28',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_11_25',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_33',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_18-11-2023_01_16',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_18-11-2023_01_19',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_56',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_13_02',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_12_14',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_20',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_17-11-2023_18_00',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_17-11-2023_20_30',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_00_40',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_00_41',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_22-11-2023_16_03',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_18',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_07-12-2023_23_54',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_07-12-2023_23_19',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_28',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_28-11-2023_20_40',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_28-11-2023_20_27',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_36',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_26-11-2023_22_27',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_26-11-2023_22_26',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_34',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_16_39',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_16_54',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_44',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_14_(1)',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_08',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_13',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_12_40',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_12_35',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_23_00',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_18-11-2023_06_58_(1)',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_18-11-2023_06_58',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_55',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_22_42',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_22_59',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_30',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_16-11-2023_21_29',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_16-11-2023_20_37',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_08-10-2023_22_02',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_08-12-2023_05_21',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_08-12-2023_05_10',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_04_10',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_07-12-2023_15_06',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_07-12-2023_12_41',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_11_06',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_03_47',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_03_46',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_34',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-11-2023_03_52',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-11-2023_03_56',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_13_19',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_09',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_19_(1)',
                     ],
        }
    },
}

ROT_4C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_22',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_17_(1)',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_21',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_04',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_11_31',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_11_15',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_22',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_20-11-2023_02_18',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_20-11-2023_02_11',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_35',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_13_30',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_13_17',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_43',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_17-11-2023_23_01',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_17-11-2023_22_24',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_00_57',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-11-2023_23_56',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_14-12-2023_02_50',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_18_(1)',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_08-12-2023_00_31',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_08-12-2023_00_37',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_05',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_28-11-2023_20_11',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_28-11-2023_20_01',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_27',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_26-11-2023_22_12',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_26-11-2023_22_22',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_29',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_16_56',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_16_51',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_38',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_14',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_07',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_15',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_12_41',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_12_36',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_04',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_20-11-2023_07_45',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_20-11-2023_07_49',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_27',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_21_57',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_30-11-2023_02_21',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_24',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_16-11-2023_22_27',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_17-11-2023_10_04',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_23_19',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_20-12-2023_23_35',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_20-12-2023_23_35_(1)',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_44',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_07-12-2023_14_05',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_07-12-2023_10_20',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_10_35',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_03_47_(1)',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_03_44',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_26',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-11-2023_03_37',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-11-2023_04_02',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_54',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_04',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_15',
                     ],
        }
    },
}

SWAP_2C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_20',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_20_(1)',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_20',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_01',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_11_33',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_25-11-2023_18_11',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_20',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-11-2023_03_05',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-11-2023_02_50',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_52',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_13_29',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_30-11-2023_13_18',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_38',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_22-11-2023_03_05',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_22-11-2023_03_09',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_29-09-2023_01_05',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_22-11-2023_16_32',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_14-12-2023_02_17',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_03',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_08-12-2023_00_04',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_08-12-2023_00_20',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_08',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_28-11-2023_20_33',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_28-11-2023_20_36',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_25',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_26-11-2023_22_38',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_26-11-2023_22_35',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_27',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_17_05',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_17_06',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_22',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_04',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_08_(1)',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_06',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_12_44',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_12_51',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_14',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_22-11-2023_08_26',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_22-11-2023_08_31',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_17',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_20_09',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_30-11-2023_04_40',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_13',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_21-11-2023_03_47',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_21-11-2023_03_46',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_23_25',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_11-01-2024_21_54',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_11-01-2024_22_40',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_52',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_07-12-2023_16_18',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_07-12-2023_15_16',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_26-09-2023_11_05',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_03_48',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_29-11-2023_06_13',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_23',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-11-2023_03_53',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-11-2023_04_01',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_49',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_19_(2)',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_18',
                     ],
        }
    },
}

SWAP_4C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_10',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_17',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_17-11-2023_01_22',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_22-09-2023_00_06',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_25-11-2023_18_10',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_25-11-2023_18_08',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_06',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_25-11-2023_04_18',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_25-11-2023_04_16',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_39',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_02-12-2023_01_36_(1)',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_02-12-2023_01_36',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_09',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_22-11-2023_03_17',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_22-11-2023_02_36',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_01_45',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_02-01-2024_17_57',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_02-01-2024_18_18',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_47',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_13-12-2023_00_07',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_13-12-2023_00_20',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_06_07',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_29-11-2023_22_56',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_29-11-2023_22_57',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_21',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-12-2023_01_37',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-12-2023_01_34',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_29_(1)',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_17_00',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_17-11-2023_17_06_(1)',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_28',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_02',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_18-11-2023_04_30',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_13_54',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_25-11-2023_21_36',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_25-11-2023_21_47',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_18',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_25-11-2023_10_45',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_25-11-2023_10_50',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_08_05',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_02-12-2023_01_51',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_02-12-2023_01_42',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_19_12',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_21-11-2023_03_24',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_21-11-2023_04_38',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_10-10-2023_22_43',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_27-12-2023_17_46',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_27-12-2023_16_31',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_10_42',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_13-12-2023_04_55',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_13-12-2023_04_27',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-09-2023_14_38',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-11-2023_03_05_(1)',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-11-2023_03_05',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_24',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_01-12-2023_03_54',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_01-12-2023_03_50',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_02-10-2023_12_54_(1)',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_19',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_19-11-2023_01_12',
                     ],
        }
    },
}

PART_2C = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_21-09-2023_23_16',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_20-11-2023_20_56',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_20-11-2023_20_32',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_05',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_06_12',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_06_23',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_23_17',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_25-11-2023_03_55',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_25-11-2023_04_03',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_36',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_02-12-2023_01_08',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_02-12-2023_01_05',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_15_21',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_25-11-2023_15_39',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_29-11-2023_02_11',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_02_24',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_28-12-2023_23_19',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_28-12-2023_22_29',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_02_10',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_13-12-2023_00_17',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_13-12-2023_00_39',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_58',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_16_23_(1)',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_16_15',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_08',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_00_01',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_00_03',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_18_24',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_21-11-2023_06_16_(2)',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_21-11-2023_06_16',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'threshold': 2,
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_13_44',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-11-2023_09_49',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-11-2023_11_28',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_13_05',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_00_37',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_22-11-2023_23_42',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_33',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_28-11-2023_10_42',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_28-11-2023_10_50',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_52',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_02-12-2023_01_00',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_02-12-2023_00_57',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_11_30',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_24-11-2023_11_53',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_27-11-2023_23_16',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_00_53',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_05-01-2024_06_51',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_05-01-2024_07_22',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_09_58',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_13-12-2023_04_55_(1)',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_13-12-2023_05_11',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_27-09-2023_14_45',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_25-11-2023_00_09',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_25-11-2023_00_03',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_18',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_04_55',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_05_08',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_05_24',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_11_54',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_11_51',
                     ],
        }
    },
}

PRACT = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'threshold': 2,
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_07',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_20-11-2023_20_37',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_20-11-2023_20_44',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_57_(1)',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_06_27',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_06_32',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_22_48',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_30-11-2023_06_05',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_30-11-2023_05_59',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_15',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_15-12-2023_22_58',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_03-12-2023_18_05',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_17_00',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_29-11-2023_04_49',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_25-11-2023_19_53',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_07-10-2023_01_32',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_12-01-2024_03_46',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_12-01-2024_03_25',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_42',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_12-12-2023_23_56',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_13-12-2023_00_19',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_37',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_16_23',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_16_18',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_19_01',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_00_53',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_01_05',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_17_57',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_21-11-2023_06_17',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_21-11-2023_06_16_(1)',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'threshold': 2,
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_16',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-11-2023_11_59',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-11-2023_09_38',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_03',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_01_28',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_00_56',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_05',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_30-11-2023_11_45',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_30-11-2023_11_42',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_27',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_20_11',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_20_12',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_18_52',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_28-11-2023_08_14',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_28-11-2023_08_10',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_04_53',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_22-12-2023_13_03',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_22-12-2023_13_04',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_09_49',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_14-12-2023_11_36',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_14-12-2023_11_35',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_06',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_25-11-2023_00_04',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_25-11-2023_00_01',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_12',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_05_19',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_05_20',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_04_58',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_12_04',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_12_03',
                     ],
        }
    },
}

PATHO = {
    'Train size=400': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'threshold': 4,
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_22-09-2023_00_02',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_20-11-2023_20_17',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse_clusters(directed-3)_20-11-2023_20_44_(1)',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_21-09-2023_23_57',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_05_56',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_06_34',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_22-09-2023_22_55',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_28-11-2023_00_19',
                       'conns/cifar10/GossipPullAgent_100A_500E_32B_aucccr(directed-3)_28-11-2023_00_20',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_23-09-2023_01_14',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_02-12-2023_01_26',
                    'conns/cifar10/DacAgent_100A_500E_32B_sparse(directed-3)_02-12-2023_01_20',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_06-10-2023_18_02',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_25-11-2023_18_19',
                          'conns/cifar10/DCliqueAgent_100A_500E_32B_d-cliques(directed-3)_25-11-2023_18_16',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_03-10-2023_02_43',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_30-12-2023_01_59',
                      'conns/cifar10/DiPLeAgent_100A_500E_32B_sparse(directed-3)_05-01-2024_20_12',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_25-09-2023_01_40',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_12-12-2023_23_56',
                    'conns/cifar10/L2CAgent_100A_500E_32B_sparse(directed-3)_13-12-2023_00_19',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_05_27',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_15_56',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_24-11-2023_16_05',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_30-09-2023_18_53',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_00_13',
                         'conns/cifar10/PanmAgent_100A_500E_32B_sparse(directed-3)_23-11-2023_00_20',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_01-10-2023_17_59',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_21-11-2023_06_12_(1)',
                     'conns/cifar10/PensAgent_100A_500E_32B_sparse(directed-3)_21-11-2023_06_12',
                     ],
        }
    },
    'Train size=100': {
        'x_axis': 'epoch',
        'metric': 'test_model-sparse_categorical_accuracy',
        'threshold': 3,
        'viz': {
            'Oracle': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_14',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-11-2023_13_14',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-11-2023_11_32',
                       ],
            'Sparse': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_20-09-2023_14_10',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_02_54',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_01_12',
                       ],
            'AUCCCR': ['conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_23-09-2023_22_24',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_28-11-2023_10_53',
                       'conns/cifar10/GossipPullAgent_100A_1000E_32B_aucccr(directed-3)_28-11-2023_10_46',
                       ],
            'DAC': ['conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_24-09-2023_07_24',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_15-12-2023_03_02',
                    'conns/cifar10/DacAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_20_02',
                    ],
            'D-Cliques': ['conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_06-10-2023_18_56',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_28-11-2023_08_19',
                          'conns/cifar10/DCliqueAgent_100A_1000E_32B_d-cliques(directed-3)_24-11-2023_20_46',
                          ],
            'DiPLe': ['conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_15-10-2023_03_17',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_22-12-2023_12_41',
                      'conns/cifar10/DiPLeAgent_100A_1000E_32B_sparse(directed-3)_30-12-2023_17_17',
                      ],
            'L2C': ['conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_25-09-2023_03_33',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_14-12-2023_11_31',
                    'conns/cifar10/L2CAgent_100A_1000E_32B_sparse(directed-3)_14-12-2023_11_37',
                    ],
            'PANMGrad': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_28-09-2023_06_15',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_23_59',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_24-11-2023_23_57',
                         ],
            'PANMLoss': ['conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_30-09-2023_05_09',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_05_20_(1)',
                         'conns/cifar10/PanmAgent_100A_1000E_32B_sparse(directed-3)_23-11-2023_05_10',
                         ],
            'PENS': ['conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-10-2023_05_04',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_11_46',
                     'conns/cifar10/PensAgent_100A_1000E_32B_sparse(directed-3)_03-12-2023_11_40',
                     ],
        }
    },
}


def plot_cifar_2C_rot():
    # Rotations = [0, 180]
    plot_results({'Train size=400': 'test_model-sparse_categorical_accuracy'}, ROT_2C['Train size=400']['viz'])
    plot_results({'Train size=100': 'test_model-sparse_categorical_accuracy'}, ROT_2C['Train size=100']['viz'])


def plot_cifar_4C_rot():
    # Rotations = [0, 90, 180, 270]
    plot_results({'Train size=400': 'test_model-sparse_categorical_accuracy'}, ROT_4C['Train size=400']['viz'])
    plot_results({'Train size=100': 'test_model-sparse_categorical_accuracy'}, ROT_4C['Train size=100']['viz'])


def plot_cifar_2C_swap():
    # label_swaps = [[], [0, 2]]
    plot_results({'N=400': 'test_model-sparse_categorical_accuracy'}, SWAP_2C['Train size=400']['viz'])
    plot_results({'N=100': 'test_model-sparse_categorical_accuracy'}, SWAP_2C['Train size=100']['viz'])


def plot_cifar_4C_swap():
    # label_swaps = [[], [[0, 1]], [[2, 3]], [[4, 5]]]
    plot_results({'N=400': 'test_model-sparse_categorical_accuracy'}, SWAP_4C['Train size=400']['viz'])
    plot_results({'N=100': 'test_model-sparse_categorical_accuracy'}, SWAP_4C['Train size=100']['viz'])


def plot_cifar_2C_part():
    # label_pertitions = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]
    plot_results({'Train size=400': 'test_model-sparse_categorical_accuracy'}, PART_2C['Train size=400']['viz'])
    plot_results({'Train size=100': 'test_model-sparse_categorical_accuracy'}, PART_2C['Train size=100']['viz'], threshold=PART_2C['Train size=100']['threshold'])


def plot_cifar_pathological():
    # pathological non-IID
    plot_results({'Train size=400': 'test_model-sparse_categorical_accuracy'}, PATHO['Train size=400']['viz'], threshold=PATHO['Train size=400']['threshold'])
    plot_results({'Train size=100': 'test_model-sparse_categorical_accuracy'}, PATHO['Train size=100']['viz'], threshold=PATHO['Train size=100']['threshold'])


def plot_cifar_practical():
    # practical non-IID
    plot_results({'Train size=400': 'test_model-sparse_categorical_accuracy'}, PRACT['Train size=400']['viz'], threshold=PRACT['Train size=400']['threshold'])
    plot_results({'Train size=100': 'test_model-sparse_categorical_accuracy'}, PRACT['Train size=100']['viz'], threshold=PRACT['Train size=100']['threshold'])


REDDIT_SMALL = {
    'LoL-Politics': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_13-10-2023_13_40',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_45',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_1000E_50B_sparse(directed-3)_20-10-2023_12_39',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_05-11-2023_23_38',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_1000E_50B_sparse(directed-3)_30-11-2023_14_05',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_1000E_50B_sparse(directed-3)_10-11-2023_05_49',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_10-11-2023_00_30',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_13-11-2023_11_46',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_1000E_50B_sparse(directed-3)_18-11-2023_09_44',
                     ],
        }
    },
    'LoL-NBA': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_15',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_46',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_1000E_50B_sparse(directed-3)_20-10-2023_12_21',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_05-11-2023_23_54',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_400E_50B_sparse(directed-3)_19-12-2023_15_51',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_1000E_50B_sparse(directed-3)_08-11-2023_08_29',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_11-11-2023_19_29',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_13-11-2023_11_10_(1)',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_1000E_50B_sparse(directed-3)_18-11-2023_13_34',
                     ],
        }
    },
    'LoL-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_12',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_25',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_1000E_50B_sparse(directed-3)_20-10-2023_11_47',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_05-11-2023_21_51',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_400E_50B_sparse(directed-3)_19-12-2023_16_17',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_1000E_50B_sparse(directed-3)_08-11-2023_11_36',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_11-11-2023_19_30',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_13-11-2023_11_10',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_1000E_50B_sparse(directed-3)_19-11-2023_21_40',
                     ],
        }
    },
    'Politics-NBA': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_10',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_15_25_(1)',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_1000E_50B_sparse(directed-3)_20-10-2023_11_55',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_28-10-2023_06_16',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_400E_50B_sparse(directed-3)_19-12-2023_17_42',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_1000E_50B_sparse(directed-3)_08-11-2023_08_06',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_11-11-2023_19_41',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_15-11-2023_05_58',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_1000E_50B_sparse(directed-3)_19-11-2023_17_40',
                     ],
        }
    },
    'Politics-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_21_02',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_14_42',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_1000E_50B_sparse(directed-3)_20-10-2023_03_27',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_28-10-2023_01_57',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_400E_50B_sparse(directed-3)_19-12-2023_17_33',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_1000E_50B_sparse(directed-3)_08-11-2023_10_35',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_12-11-2023_22_24',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_15-11-2023_05_05',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_1000E_50B_sparse(directed-3)_20-11-2023_21_24',
                     ],
        }
    },
    'NBA-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse_clusters(directed-3)_16-10-2023_20_57',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_1000E_50B_sparse(directed-3)_18-10-2023_14_39',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_1000E_50B_sparse(directed-3)_20-10-2023_03_28',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_1000E_50B_d-cliques(directed-3)_28-10-2023_01_35',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_400E_50B_sparse(directed-3)_25-12-2023_05_13',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_1000E_50B_sparse(directed-3)_10-11-2023_04_41',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_11-11-2023_21_22',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_1000E_50B_sparse(directed-3)_15-11-2023_04_45',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_1000E_50B_sparse(directed-3)_21-11-2023_04_00',
                     ],
        }
    },
    'LoL-Politics-NBA-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_200A_1000E_50B_sparse_clusters(directed-3)_18-10-2023_14_27',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_200A_1000E_50B_sparse(directed-3)_20-10-2023_21_42',
                       ],
            'DAC': ['conns/reddit/DacAgent_200A_1000E_50B_sparse(directed-3)_21-10-2023_16_51',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_200A_1000E_50B_d-cliques(directed-3)_13-11-2023_12_50',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_200A_1000E_50B_sparse(directed-3)_15-12-2023_12_24',
                      ],
            'L2C': ['conns/reddit/L2CAgent_200A_1000E_50B_sparse(directed-3)_26-11-2023_15_08',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_200A_1000E_50B_sparse(directed-3)_21-11-2023_22_14',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_200A_1000E_50B_sparse(directed-3)_16-11-2023_23_33',
                         ],
            'PENS': ['conns/reddit/PensAgent_200A_1000E_50B_sparse(directed-3)_23-11-2023_23_58',
                     ],
        }
    }
}


def plot_reddit_small():
    # LoL <-> Politics
    plot_results({'LoL-Politics': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['LoL-Politics']['viz'], analyze='gini')

    # LoL <-> NBA
    plot_results({'LoL-NBA': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['LoL-NBA']['viz'], analyze='gini')

    # LoL <-> Bitcoin
    plot_results({'LoL-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['LoL-Bitcoin']['viz'], analyze='gini')

    # Politics <-> NBA
    plot_results({'Politics-NBA': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['Politics-NBA']['viz'], analyze='gini')

    # Politics <-> Bitcoin
    plot_results({'Politics-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['Politics-Bitcoin']['viz'], analyze='gini')

    # NBA <-> Bitcoin
    plot_results({'NBA-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['NBA-Bitcoin']['viz'], analyze='gini')

    # LoL-Politics-NBA-Bitcoin
    plot_results({'LoL-Politics-NBA-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_SMALL['LoL-Politics-NBA-Bitcoin']['viz'], analyze='gini')


REDDIT_LARGE = {
    'LoL-Politics': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_22-12-2023_06_48',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_05-01-2024_20_04',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_06-01-2024_02_39',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_16-12-2023_15_06',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_27-01-2024_15_56',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_27-01-2024_21_36',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_300E_50B_sparse(directed-3)_22-12-2023_14_05',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_300E_50B_d-cliques(directed-3)_29-12-2023_14_05',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_300E_50B_sparse(directed-3)_28-01-2024_19_32',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_300E_50B_sparse(directed-3)_24-12-2023_09_32',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_01-01-2024_03_28',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_28-12-2023_08_27',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_300E_50B_sparse(directed-3)_07-01-2024_17_55',
                     ],
        }
    },
    'LoL-NBA': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_22-12-2023_02_58',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_05-01-2024_23_52',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_14-01-2024_07_36',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_16-12-2023_16_48',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_27-01-2024_20_19',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_27-01-2024_17_31',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_300E_50B_sparse(directed-3)_22-12-2023_11_41',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_300E_50B_d-cliques(directed-3)_30-12-2023_04_53',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_300E_50B_sparse(directed-3)_28-01-2024_12_38',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_300E_50B_sparse(directed-3)_25-12-2023_09_20',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_02-01-2024_08_44',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_28-12-2023_09_16',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_300E_50B_sparse(directed-3)_08-01-2024_18_42',
                     ],
        }
    },
    'LoL-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_21-12-2023_23_55',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_14-01-2024_08_33',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_14-01-2024_04_58',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_17-12-2023_14_30',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_31-01-2024_18_24',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_31-01-2024_16_01',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_300E_50B_sparse(directed-3)_22-12-2023_01_18',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_300E_50B_d-cliques(directed-3)_03-01-2024_11_51',
                          ],
            'DiPLe': ['conns/reddit/DiPLeAgent_100A_120E_50B_sparse(directed-3)_07-02-2024_10_37',
                      ],
            'L2C': ['conns/reddit/L2CAgent_100A_300E_50B_sparse(directed-3)_24-12-2023_15_24',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_03-01-2024_06_41',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_29-12-2023_09_40',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_300E_50B_sparse(directed-3)_09-01-2024_16_12',
                     ],
        }
    },
    'Politics-NBA': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_26-12-2023_19_01',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_14-01-2024_11_01',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_14-01-2024_14_56',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_17-12-2023_18_38',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_31-01-2024_20_39',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_01-02-2024_00_14',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_300E_50B_sparse(directed-3)_21-12-2023_23_43',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_300E_50B_d-cliques(directed-3)_06-01-2024_18_02',
                          ],
            # 'DiPLe': ['conns/reddit/',
            #           ],
            'L2C': ['conns/reddit/L2CAgent_100A_300E_50B_sparse(directed-3)_25-12-2023_23_13',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_07-01-2024_15_13',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_29-12-2023_13_12',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_300E_50B_sparse(directed-3)_10-01-2024_13_32',
                     ],
        }
    },
    'Politics-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_27-12-2023_15_22',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_17-01-2024_23_35',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_18-01-2024_05_19',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_20-12-2023_16_57',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_04-02-2024_21_40',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_05-02-2024_03_52',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_300E_50B_sparse(directed-3)_21-12-2023_07_49',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_300E_50B_d-cliques(directed-3)_07-01-2024_10_56',
                          ],
            # 'DiPLe': ['conns/reddit/',
            #           ],
            'L2C': ['conns/reddit/L2CAgent_100A_300E_50B_sparse(directed-3)_26-12-2023_10_44',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_08-01-2024_09_42',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_30-12-2023_20_13',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_300E_50B_sparse(directed-3)_09-01-2024_13_53',
                     ],
        }
    },
    'NBA-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_26-12-2023_02_05',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_18-01-2024_04_12',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse_clusters(directed-3)_18-01-2024_00_30',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_20-12-2023_13_55',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_05-02-2024_01_47',
                       'conns/reddit/GossipPullAgent_100A_300E_50B_sparse(directed-3)_04-02-2024_22_30',
                       ],
            'DAC': ['conns/reddit/DacAgent_100A_300E_50B_sparse(directed-3)_21-12-2023_05_00',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_100A_120E_50B_d-cliques(directed-3)_04-02-2024_08_10',
                          ],
            # 'DiPLe': ['conns/reddit/',
            #           ],
            'L2C': ['conns/reddit/L2CAgent_100A_300E_50B_sparse(directed-3)_27-12-2023_04_21',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_08-01-2024_10_19',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_100A_300E_50B_sparse(directed-3)_30-12-2023_20_29',
                         ],
            'PENS': ['conns/reddit/PensAgent_100A_300E_50B_sparse(directed-3)_09-01-2024_12_07',
                     ],
        }
    },
    'LoL-Politics-NBA-Bitcoin': {
        'x_axis': 'epoch',
        'metric': 'test_model-accuracy_no_oov',
        'viz': {
            'Oracle': ['conns/reddit/GossipPullAgent_200A_300E_50B_sparse_clusters(directed-3)_31-08-2023_22_15',
                       'conns/reddit/GossipPullAgent_200A_300E_50B_sparse_clusters(directed-3)_02-01-2024_06_44',
                       'conns/reddit/GossipPullAgent_200A_300E_50B_sparse_clusters(directed-3)_02-01-2024_11_48',
                       ],
            'Sparse': ['conns/reddit/GossipPullAgent_200A_300E_50B_sparse(directed-3)_21-12-2023_17_46',
                       'conns/reddit/GossipPullAgent_200A_300E_50B_sparse(directed-3)_23-01-2024_18_56',
                       'conns/reddit/GossipPullAgent_200A_300E_50B_sparse(directed-3)_23-01-2024_20_12',
                       ],
            'DAC': ['conns/reddit/DacAgent_200A_300E_50B_sparse(directed-3)_23-12-2023_00_54',
                    ],
            'D-Cliques': ['conns/reddit/DCliqueAgent_200A_90E_50B_d-cliques(directed-3)_04-02-2024_07_44',
                          ],
            'DiPLe': ['DiPLeAgent_200A_100E_50B_sparse(directed-3)_06-02-2024_17_04',
                      ],
            'L2C': ['L2CAgent_200A_76E_50B_sparse(directed-3)_06-02-2024_18_48',
                    ],
            'PANMGrad': ['conns/reddit/PanmAgent_200A_300E_50B_sparse(directed-3)_01-02-2024_21_35',
                         ],
            'PANMLoss': ['conns/reddit/PanmAgent_200A_300E_50B_sparse(directed-3)_02-01-2024_22_24',
                         ],
            'PENS': ['conns/reddit/PensAgent_200A_300E_50B_sparse(directed-3)_11-01-2024_16_35',
                     ],
        }
    }
}


def plot_reddit_large():
    # LoL <-> Politics
    plot_results({'LoL-Politics': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['LoL-Politics']['viz'], analyze='gini')

    # LoL <-> NBA
    plot_results({'LoL-NBA': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['LoL-NBA']['viz'], analyze='gini')

    # LoL <-> Bitcoin
    plot_results({'LoL-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['LoL-Bitcoin']['viz'], analyze='gini')

    # Politics <-> NBA
    plot_results({'Politics-NBA': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['Politics-NBA']['viz'], analyze='gini')

    # Politics <-> Bitcoin
    plot_results({'Politics-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['Politics-Bitcoin']['viz'], analyze='gini')

    # NBA <-> Bitcoin
    plot_results({'NBA-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['NBA-Bitcoin']['viz'], analyze='gini')

    # LoL-Politics-NBA-Bitcoin
    plot_results({'LoL-Politics-NBA-Bitcoin': 'test_model-accuracy_no_oov'}, REDDIT_LARGE['LoL-Politics-NBA-Bitcoin']['viz'], analyze='gini')


def plot_results(info, data, n_rows=1, axis_lim=None, title=None, metric='avg', analyze='accs', threshold=1):
    # analyze in [all, plot, accs, table, msgs, gini]
    viz = {
        lbl: {
            'x_axis': 'epoch',
            'metric': d_metric,
            'viz': data
        } for lbl, d_metric in info.items()
    }
    if analyze in ['all', 'plot']:
        side_by_side(viz, n_rows=n_rows, axis_lim=axis_lim)
    # """
    if analyze in ['all', 'accs']:
        for vk, vv in viz.items():
            print(vk)
            for k, v in vv['viz'].items():
                avg_agents_max = get_max_avg_acc(v, vv['metric'])
                stable_avg, start, end, t = find_max_avg_acc(v, vv['metric'], threshold=threshold)

                print('\t-', k.ljust(10),
                      round(max(t), 2), f"({round(stable_avg, 2)}% - {start}:{end})".ljust(13),
                      "\tAvg-Agent-Max", str(round(float(avg_agents_max), 2)).ljust(5))

    if analyze in ['all', 'table']:
        create_table(viz, title=title, metric=metric, print_means=False)

    if analyze in ['all', 'msgs']:
        # plot_comms(viz)
        print("==== Messages ====\n\t\tReceived\tSent")
        for vk, vv in viz.items():
            for k, v in vv['viz'].items():
                print(k)
                msgs(v, verbose=2)

    if analyze in ['all', 'gini']:
        print("=== GINI ===")
        for vk, vv in viz.items():
            print(vk)
            for k, v in vv['viz'].items():
                gini_coef, avg_msgs = msgs(v, verbose=0)
                print('\t', k.ljust(10), "{:.3f} ({})".format(gini_coef, round(avg_msgs)))


def get_max_avg_acc(paths, metric):
    accs = parse_timeline(None, paths, x_axis='examples', metric=metric, agg_fn=np.array)[2]
    # avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs]))
    avg_agents_max = np.mean([np.max(np.mean(sim, axis=1), axis=0) for sim in accs])
    # t_acc = parse_timeline(None, paths, x_axis='examples', metric=metric, agg_fn=np.mean)[1]
    # avg_agents_max = max(t_acc)
    return avg_agents_max


def find_max_avg_acc(paths, metric, threshold=1):
    t = parse_timeline(None, paths, x_axis='examples', metric=metric)[1]
    # start, end = identify_stable_region(t)
    start, end = find_convergence_region(t, threshold=threshold)
    return max(t[start:end]), start, end, t


def build_acc_msg_data(viz):
    names = list(viz[list(viz.keys())[0]]['viz'].keys())
    data = {n: ([], []) for n in names}
    for k, v in viz.items():
        for kv, vv in v['viz'].items():
            acc, start, end, t = find_max_avg_acc(vv, v['metric'], threshold=v.get('threshold', 1))
            iteration = np.argwhere(t[start:end] == acc).flatten()[0] + start

            messages = []
            for p in vv:
                sent_msgs = []
                agents = read_agents(p)
                for a_k, a_v in agents.items():
                    sent_msgs.append(sum(a_v['sent_msg'][:iteration]))
                messages.append(np.mean(sent_msgs))

            data[kv][0].append(acc)
            data[kv][1].append(np.mean(messages))

    for dk in data.keys():
        data[dk] = [np.mean(data[dk][0]), np.mean(data[dk][1])]
    return data


def pareto_front(viz, title, fig_name=None):
    if isinstance(viz, (list, tuple)):
        print(f"Pareto-front: parsing {len(viz)} groups")
        d_list = [build_acc_msg_data(v) for v in viz]
        data = {}
        for d in d_list:
            for dk, dv in d.items():
                if dk not in data:
                    data[dk] = []
                data[dk].append(dv)
        for dk in data.keys():
            data[dk] = [np.mean([v[0] for v in data[dk]]), np.mean([v[1] for v in data[dk]])]
    else:
        data = build_acc_msg_data(viz)
    # sorted_data = data
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1][0], reverse=True))

    def is_dominated(point, point_data):
        for other_point in point_data:
            if (
                    other_point[0] >= point[0] and other_point[1] <= point[1]
                    and (other_point[0] > point[0] or other_point[1] < point[1])
            ):
                return False
        return True

    pf = [point for point in list(sorted_data.values()) if is_dominated(point, list(sorted_data.values()))]
    pareto_y, pareto_x = zip(*pf)

    plt.rcParams.update({'font.size': 14})
    plt.plot(pareto_x, pareto_y, c='lightgray', label='Pareto front', marker='o', markersize=15, linewidth=2, zorder=-1)

    marker_styles = ['o', 'x', 's', 'D', '^', 'v', '>', '<', 'p', 'H']
    for i, (k, v) in enumerate(data.items()):
        plt.scatter(v[1], v[0], marker=marker_styles[i], s=80, label=k)

    plt.ylabel('Accuracy')
    plt.xlabel('Number of messages per agent')
    if title:
        plt.title(title)

    plt.legend() # loc='upper left')
    plt.gca().invert_xaxis()
    plt.xscale('log')
    # plt.gcf().set_size_inches(13, 10)

    # Set custom x-axis ticks (adjust the range as needed)
    # x = [v[1] for v in sorted_data.values()]
    # x_ticks = np.logspace(np.log10(max(x)), np.log10(min(x)), num=2)
    # x_labels = ['1e{0}'.format(int(np.log10(x))) for x in x_ticks]
    # plt.xticks(x_ticks, x_labels)
    plt.grid(True)
    plt.tight_layout()

    if fig_name:
        plt.savefig(f"plot/plots/{fig_name}.pdf")
        plt.clf()
    else:
        plt.show()


def plot_cifar_paretos():
    pareto_front(ROT_2C, None, 'rot_2c')  # 'Rotation=$\{0\degree, 180\degree\}$'
    pareto_front(ROT_4C, None, 'rot_4c')  # 'Rotation=$\{0\degree, 90\degree, 180\degree, 270\degree\}$'

    pareto_front(SWAP_2C, None, 'swap_2c')  # 'Swap={None, [0, 2]}'
    pareto_front(SWAP_4C, None, 'swap_4c')  # 'Swap={None, [0, 1], [2, 3], [4, 5]}'

    pareto_front(PART_2C, None, 'part_2c')  # 'Part={Vehicles, Animals}'
    pareto_front(PRACT, None, 'part_pract')  # 'Practical non-IID'
    pareto_front(PATHO, None, 'part_patho')  # 'Pathological non-IID'

    dicts = [ROT_2C, ROT_4C, SWAP_2C, SWAP_4C, PART_2C, PRACT, PATHO]
    all_d = {}
    for di, d in enumerate(dicts):
        all_d.update({f'dict{di}_{key}': value for key, value in d.items()})
    pareto_front(all_d, None, 'cifar_mean')  # 'Mean across experiments'


def plot_reddit_paretos():
    def two_cluster(reddit_dict):
        c = reddit_dict.copy()
        del c['LoL-Politics-NBA-Bitcoin']
        return c

    pareto_front(two_cluster(REDDIT_SMALL), None, "04_r_small_2c")
    pareto_front({'LoL-Politics-NBA-Bitcoin': REDDIT_SMALL['LoL-Politics-NBA-Bitcoin']}, None, "04_r_small_4c")

    pareto_front(two_cluster(REDDIT_LARGE), None, "04_r_large_2c")
    pareto_front({'LoL-Politics-NBA-Bitcoin': REDDIT_LARGE['LoL-Politics-NBA-Bitcoin']}, None, "04_r_large_4c")
    """
    dicts = [# two_cluster(REDDIT_SMALL), two_cluster(REDDIT_LARGE),
             {'LoL-Politics': REDDIT_SMALL['LoL-Politics']},
             {'LoL-Politics': REDDIT_LARGE['LoL-Politics']},
             {'LoL-Politics-NBA-Bitcoin': REDDIT_SMALL['LoL-Politics-NBA-Bitcoin']},
             {'LoL-Politics-NBA-Bitcoin': REDDIT_LARGE['LoL-Politics-NBA-Bitcoin']}]
    all_d = {}
    for di, d in enumerate(dicts):
        all_d.update({f'dict{di}_{key}': value for key, value in d.items()})
    pareto_front(all_d, None, 'reddit_mean')  # 'Mean across experiments'
    """
    dicts = [[two_cluster(REDDIT_SMALL), two_cluster(REDDIT_LARGE)],
             [{'LoL-Politics-NBA-Bitcoin': REDDIT_SMALL['LoL-Politics-NBA-Bitcoin']},
             {'LoL-Politics-NBA-Bitcoin': REDDIT_LARGE['LoL-Politics-NBA-Bitcoin']}]]
    all_d = [{} for _ in range(len(dicts))]
    for i in range(len(all_d)):
        for di, d in enumerate(dicts[i]):
            all_d[i].update({f'dict{di}_{key}': value for key, value in d.items()})
    pareto_front(all_d, None, '04_reddit_mean')  # 'Mean across experiments'


def _reddit_results():
    for key in REDDIT_LARGE.keys():
        viz = {key: REDDIT_LARGE[key]}
        v = viz[key]['viz']
        viz[key]['viz'] = {
            vk: vv for vk, vv in v.items() if not vv[0].endswith('conns/reddit/')
        }
        data = build_acc_msg_data(viz)
        print(key)
        for dk, dv in data.items():
            print(f'\t- {dk}'.ljust(20), "{:.3}".format(dv[0]))


def reddit_grouped_results():
    viz_dict = REDDIT_LARGE
    keys = viz_dict[list(viz_dict.keys())[0]]['viz'].keys()
    accs_2c = {k: [] for k in keys}
    print("4 clusters")
    for key in viz_dict.keys():
        data = build_acc_msg_data({key: viz_dict[key]})
        for dk, dv in data.items():
            if key.count('-') == 3:
                print(f'\t- {dk}'.ljust(20), "{:.3}".format(dv[0]))
            else:
                accs_2c[dk].append(dv[0])

    print("2 clusters")
    for k, v in accs_2c.items():
        print(f'\t- {k}'.ljust(20), "{:.3}".format(np.mean(v)))


def find_longest_consecutive_streak(numbers):
    if numbers is None or len(numbers) == 0:
        return []

    longest_streak = []
    current_streak = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_streak.append(numbers[i])
        else:
            if len(current_streak) > len(longest_streak):
                longest_streak = current_streak.copy()
            current_streak = [numbers[i]]

    # Check if the last streak is longer than the longest streak
    if len(current_streak) > len(longest_streak):
        longest_streak = current_streak

    return longest_streak


def identify_stable_region(values, threshold=1):
    # Calculate the derivative
    accuracy_derivative = np.diff(values)

    # Find the stable region indices where the derivative is close to zero
    stable_region_indices = np.where(np.abs(accuracy_derivative) < threshold)[0] + 1  # Add 1 to align with epoch numbers
    stable_region_indices = find_longest_consecutive_streak(list(stable_region_indices))

    """
    print("Stable Region Indices:", stable_region_indices)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    # Highlight the stable region
    plt.axvline(x=stable_region_indices[0], color='red', linestyle='--', label='Stable Region Start')
    plt.axvline(x=stable_region_indices[-1], color='green', linestyle='--', label='Stable Region End')
    plt.legend()
    plt.show()
    """
    return stable_region_indices[0], stable_region_indices[-1]


def find_convergence_region(accuracy_values, threshold=0.001):
    n = len(accuracy_values)

    # Calculate the standard deviation for all possible window sizes
    indices = []
    for w_size in range(300, 30, -10):
        for i in range(w_size, n):
            vals = accuracy_values[i - w_size:i]
            # if np.std(vals) < threshold:
            if max(vals) - min(vals) < threshold:
                indices.append([i - w_size, i])
        if len(indices) > 0 and indices[0][-1] - indices[0][0] > w_size * 1.33:
            break

    # Find the longest consecutive region
    longest_indices = max(indices, key=lambda x: max(accuracy_values[x[0]:x[-1]]), default=[])

    # Find the beginning and end of the longest consecutive region
    if longest_indices:
        convergence_start = longest_indices[0]
        convergence_end = longest_indices[-1]
        return convergence_start, convergence_end
    else:
        # If no convergence region is found, return None
        return None


def msgs(path, verbose=0):
    from plot.visualize import read_agents
    if verbose:
        print('ID\tReceived\tSent')
    rec, sent = [], []
    if isinstance(path, str):
        path = [path]
    all_vals = {'gini': [], 'sent': []}
    for p in path:
        agents = read_agents(p)
        for k, v in agents.items():
            if verbose > 2:
                print(k, '\t', sum(v['useful_msg']), '\t', sum(v['sent_msg']))
            rec.append(sum(v['useful_msg']))
            sent.append(sum(v['sent_msg']))
        if verbose > 2:
            print("-----------")
        if verbose > 1:
            print(f"\t- Min\t{np.min(rec)}\t{np.min(sent)}")
            print(f"\t- Max\t{np.max(rec)}\t{np.max(sent)}")
            print(f"\t- Avg\t{np.mean(rec)}+-{np.std(rec).round()}\t{np.mean(sent)}+-{np.std(sent).round()}")
            print(f"\t- Gini\t{gini(sent)}")
        all_vals['gini'].append(gini(sent))
        all_vals['sent'].append(np.mean(sent))

    return float(np.mean(all_vals['gini'])), float(np.mean(all_vals['sent']))


def plot_comms(viz):
    # from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size': 20})
    for vk, vv in viz.items():
        for k, v in vv['viz'].items():
            # v = ['conns/cifar10/GossipPullAgent_100A_1000E_32B_sparse_clusters(directed-3)_20-09-2023_14_44']
            agents = read_agents(v[0])
            mx = np.zeros((len(agents), len(agents)))
            for i in range(len(agents)):
                for com in agents[str(i)]['sent_to']:
                    for j in com:
                        mx[i, j] += 1
            plt.pcolormesh(mx, cmap='Blues')  # , cmap=ListedColormap(['white', 'blue']))
            plt.plot([50, 50], [0, 100], [0, 100], [50, 50], color='lightgray')
            plt.xlabel('Agent ID')
            plt.ylabel('Agent ID')
            cbar = plt.colorbar()
            cbar.set_label("# of agent sent messages")
            plt.tight_layout()
            plt.savefig(f"plot/plots/{k}.pdf")
            plt.clf()


def create_table(viz, title=None, start_epoch=20, metric='both', print_means=True):
    # Method C1-Cn Mean (std)

    print("----Table----")
    print("\\begin{table*}[b]\n\\centering\n%\\footnotesize\n\\begin{tabular}{l " + ' '.join(
        ['c'] * len(viz)) + " c}\n\\hline")

    if title:
        print("\\multicolumn{" + str(len(viz) + 2) + "}{c}{\\textbf{" + title + "}} \\\\")

    print("\\textbf{Method} & " + ' '.join(
        ["\\textbf{" + k + "} &" for k in viz.keys()]) + "\\textbf{Mean}  \\\\\n\\hline")
    seen = set()
    dist_items = [x for x in [subitem for item in [v['viz'].keys() for v in viz.values()] for subitem in item] if
                  not (x in seen or seen.add(x))]
    baselines = {}
    means = {}
    a_means = {}
    for item in dist_items:
        means[item] = []
        a_means[item] = []
        if item not in ['Oracle', 'Solo']:
            print(item, end='')
        for k, v in viz.items():
            vv = v['viz'][item]
            agents_max, max_a = 0, 0
            if metric == 'both':
                t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                max_a = round(max(t), 2)

                agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                agents_max = round(float(avg_agents_max), 2)
                # t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)
            elif metric == 'avg':
                t, accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'])[1:]
                max_a = round(max(t), 2)
            else:
                agent_accs = parse_timeline(None, vv, x_axis='examples', metric=v['metric'], agg_fn=np.array)[2]
                avg_agents_max = np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in agent_accs]))
                max_a = round(float(avg_agents_max), 2)
                t = np.mean([np.max(np.array(sim), axis=0) for sim in agent_accs], axis=-1)

            means[item].append(max_a)
            a_means[item].append(agents_max)

            if item == 'Oracle' or item == 'Solo':
                baselines['Oracle-' + k] = (t[start_epoch:], max_a)
                baselines['Oracle-a-' + k] = (t[start_epoch:], agents_max)
                continue
            elif item == 'Sparse':
                baselines[k] = (t[start_epoch:], max_a)
                num = "{:.2f}".format(round(max_a, 2))
                if agents_max > 0:
                    num += " ({:.2f})".format(round(agents_max, 2))
                print(f" & {num}", end='')
                if k == list(viz.keys())[-1]:
                    if print_means:
                        nm = "{:.2f}".format(round(float(np.mean(means[item])), 2))
                        if sum(a_means[item]) > 0:
                            nm += " ({:.2f})".format(round(float(np.mean(a_means[item])), 2))
                        print(f" & {nm}", end='')
                    print(" \\\\")
            else:
                baseline = baselines[k]
                rel_inc = round((max_a - baseline[1]) / baseline[1] * 100, 2)
                """
                p_val = ttest_ind(baseline[0], t[start_epoch:])[1]
                p_text = "{} {}".format("{:.2f}".format(max_a) + '\\%', '({}\\%)'.format('+' + "{:.2f}".format(rel_inc) if rel_inc > 0 else "{:.2f}".format(rel_inc)))
                if rel_inc > 0:
                    p_text = "\\textbf{" + p_text + "}"
                if p_val < 0.05:
                    p_text += " \\textbf{**}"
                """
                num = "{:.2f}".format(round(max_a, 2))  # + (" \\textbf{**}" if p_val<0.05 else "")
                if agents_max > 0:
                    num += " ({:.2f})".format(round(agents_max, 2))
                if rel_inc > 0:
                    num = "\\textbf{" + num + "}"
                print(f" & {num}", end='')
                if k == list(viz.keys())[-1]:
                    if print_means:
                        m = float(np.mean(means[item]))
                        nm = "{:.2f}".format(round(m, 2))
                        if sum(a_means[item]) > 0:
                            nm += " ({:.2f})".format(round(float(np.mean(a_means[item])), 2))
                        if m > float(np.mean(means['Sparse'])):
                            nm = "\\textbf{" + nm + "}"
                        print(f" & {nm}", end='')
                    print(" \\\\")

    print('\\arrayrulecolor{gray}\\hline')
    print("\\textcolor[gray]{.5} {Oracle}", end='')
    for k in viz.keys():
        num = "{:.2f}".format(round(baselines['Oracle-' + k][1], 2))
        if baselines['Oracle-a-' + k][1] > 0:
            num += " ({:.2f})".format(round(baselines['Oracle-a-' + k][1], 2))
        print(" & \\textcolor[gray]{.5} {" + str(num) + '}', end='')
        if k == list(viz.keys())[-1]:
            if print_means:
                nm = "{:.2f}".format(round(float(np.mean(means['Oracle'])), 2))
                if sum(a_means['Oracle']) > 0:
                    nm += " ({:.2f})".format(round(float(np.mean(a_means['Oracle'])), 2))
                print("& \\textcolor[gray]{.5} {" + str(nm) + "}", end='')
            print(" \\\\")
    print("\\end{tabular}\n\\caption{\\label{tbl:name} " + metric + ".}\n\\end{table*}")


def do_parse_info():
    import os, json
    fnames = [f for f in os.listdir('/Users/robert/Downloads/') if f.endswith('.json')]
    f_info = {}
    for filename in fnames:
        with open('/Users/robert/Downloads/' + filename, "r") as infile:
            f_json = json.loads(infile.read())
        info = f_json['info']
        dp = info['agent_data_pars']
        d_info = dp[dp.index('50, ') + 4:-2]
        seed = info['sim_pars']['seed']
        # print(filename, f'\n\t{d_info}\n\tseed:{seed}')
        if d_info + ", seed: " + seed in f_info:
            print(d_info + ", seed: " + seed, filename, f_info[d_info + ", seed: " + seed])
        f_info[d_info + ", seed: " + seed] = filename.replace('.json', '')

    order = ["'rotations': [0, 180]", "'rotations': [0, 180], 'samples': 100",
             "'rotations': [0, 90, 180, 270], 'clusters': 4",
             "'rotations': [0, 90, 180, 270], 'clusters': 4, 'samples': 100",
             "'label_swaps': [[], [[0, 2]]]", "'label_swaps': [[], [[0, 2]]], 'samples': 100",
             "'label_swaps': [[], [[0, 1]], [[2, 3]], [[4, 5]]], 'clusters': 4",
             "'label_swaps': [[], [[0, 1]], [[2, 3]], [[4, 5]]], 'clusters': 4, 'samples': 100",
             "'label_partitions': [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]",
             "'label_partitions': [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]], 'samples': 100",
             "'mode': 'practical non-IID'", "'mode': 'practical non-IID', 'samples': 100",
             "'mode': 'pathological non-IID'", "'mode': 'pathological non-IID', 'samples': 100",
             ]

    s_keys = sorted(f_info.keys())
    for o in order:
        for k in s_keys:
            if k[:k.index(', seed: ')] == o:
                print(k, '\n\t' + f_info[k], '\n')

    for k, v in f_info.items():
        print(k, '\n\t', v)


if __name__ == '__main__':
    pass
    # plot_cifar_2C_rot()


