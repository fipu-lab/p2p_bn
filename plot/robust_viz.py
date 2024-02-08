from plot.visualize import side_by_side, show, plot_graph, resolve_timeline, parse_timeline, read_json
import numpy as np

TM = 'test_model-sparse_categorical_accuracy'
VM = 'val_model-sparse_categorical_accuracy'
COLORS = ['green', 'blue', 'red', 'orange']


def mnist_exp_ring():
    viz = {
        'IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_28',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_28_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_35',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_28',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_28_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_35',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_22',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_14_27',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_23',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_22',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_14_27',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_23',
                ]
            },
        },
        'patološki ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_34',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_34_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_33',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_34',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_34_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_33',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_28',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_24',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_28_(1)',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_28',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_24',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_28_(1)',
                ]
            },
        },
        'praktični ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_38',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_37',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_27',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_38',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_37',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_01-01-2024_18_27',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_20',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_15',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_16',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_20',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_15',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_01-01-2024_10_16',
                ]
            },
        },
    }
    side_by_side(viz, n_rows=1, fig_size=(15, 4),
                 axis_lim=[{'y': [0, 100], 'step': 5}, {'y': [0, 100], 'step': 5}, {'y': [0, 100], 'step': 5}])
    print_accs(viz)
    
    
def mnist_exp_sparse():
    viz = {
        'IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_21',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_28',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_14',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_21',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_28',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_14',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_42_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_42_(2)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_44',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_42_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_42_(2)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_44',
                ]
            },
        },
        'patološki ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_23',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_22',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_28_(1)',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_23',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_22',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_28_(1)',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_40',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_39',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_42',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_40',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_39',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_42',
                ]
            },
        },
        'praktični ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_17',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_15',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_16',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_17',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_15',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_03-01-2024_20_16',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_38',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_38_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_37',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_38',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_38_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_02-01-2024_18_37',
                ]
            },
        },
    }
    side_by_side(viz, n_rows=1, fig_size=(15, 4),
                 axis_lim=[{'y': [0, 100], 'step': 5}, {'y': [0, 100], 'step': 5}, {'y': [0, 100], 'step': 5}])
    print_accs(viz)


def cifar_exp_ring():
    viz = {
        'IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_45',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_51',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_51_(1)',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_45',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_51',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_51_(1)',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_54',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_55',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_57',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_54',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_55',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_57',
                ]
            },
        },
        'patološki ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_47_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_21_43',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_21_58',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_47_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_21_43',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_21_58',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_04',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_11',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_05',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_04',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_11',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_05',
                ]
            },
        },
        'praktični ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_47',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_18_19',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_22_23',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_17_47',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_18_19',
                    'rob/GossipPullAgent_100A_100E_32B_ring(directed-1)_31-12-2023_22_23',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_00',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_01',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_59',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_00',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_16_01',
                    'rob/P2PAgent_100A_100E_32B_ring(directed-1)_30-12-2023_15_59',
                ]
            },
        },
    }
    side_by_side(viz, n_rows=1, fig_size=(15, 4),
                 axis_lim=[{'y': [0, 85], 'step': 5}, {'y': [0, 85], 'step': 5}, {'y': [0, 85], 'step': 5}])
    print_accs(viz)


def cifar_exp_sparse():
    viz = {
        'IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_04',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_07',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_05',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_04',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_07',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_05',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_56',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_58',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_01',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_56',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_58',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_01',
                ]
            },
        },
        'patološki ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_09',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_10',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_19_31',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_09',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_10',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_19_31',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_13',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_05',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_13_(1)',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_13',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_05',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_10_13_(1)',
                ]
            },
        },
        'praktični ne-IID': {
            'x_axis': 'epoch',
            'colors': COLORS,
            'metric': [TM, VM, TM, VM],
            'viz': {
                'Gossip-Pull (test-local)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_05_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_19',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_04_(1)',
                ],
                'Gossip-Pull (test-global)': [
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_05_(1)',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_19',
                    'rob/GossipPullAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_16_04_(1)',
                ],
                'P2P-BN (test-local)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_58_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_56_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_59',
                ],
                'P2P-BN (test-global)': [
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_58_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_56_(1)',
                    'rob/P2PAgent_100A_100E_32B_sparse(directed-3)_31-12-2023_09_59',
                ]
            },
        },
    }
    side_by_side(viz, n_rows=1, fig_size=(15, 4),
                 axis_lim=[{'y': [0, 85], 'step': 5}, {'y': [0, 85], 'step': 5}, {'y': [0, 85], 'step': 5}])
    print_accs(viz)


def get_max_avg_acc(paths, metric):
    # accs = parse_timeline(None, paths, x_axis='examples', metric=metric, agg_fn=np.array)[2]
    # avg_agents_max = np.mean([np.max(np.mean(sim, axis=1), axis=0) for sim in accs])
    # print(np.mean(np.concatenate([np.max(np.array(sim), axis=0) for sim in accs])))
    t_acc = parse_timeline(None, paths, x_axis='examples', metric=metric, agg_fn=np.mean)[1]
    avg_agents_max = max(t_acc)
    return avg_agents_max


def print_accs(viz):
    for vk, vv in viz.items():
        print(vk)
        for i, (k, v) in enumerate(vv['viz'].items()):
            avg_agents_max = get_max_avg_acc(v, vv['metric'][i])
            print('\t', str(k).ljust(25), round(avg_agents_max, 2))


