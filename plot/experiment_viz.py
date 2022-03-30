from plot.visualize import side_by_side, show, plot_graph, resolve_timeline, parse_timeline, read_json
from scipy.stats import mannwhitneyu
import numpy as np

ALG_NAME = 'P2P-BN'


def plot_graphs():
    plot_graph(
        {
            'Ring (undirected)': 'graphs/P2PAgent_10A_0E_50B_1V_ring(undirected)_N10_NB2_TV-1_04-01-2022_16_11',
            'Fully connected (undirected)': 'graphs/P2PAgent_10A_0E_50B_1V_complete(undirected)_N10_NB9_TV-1_04-01-2022_16_13',
            'Sparse (undirected)': 'graphs/P2PAgent_10A_0E_50B_1V_sparse(undirected)_N10_NB3_TV-1_04-01-2022_17_06',

            'Ring (directed)': 'graphs/P2PAgent_10A_0E_50B_1V_ring(directed)_N10_NB1_TV-1_04-01-2022_17_17',
            'Fully connected (directed)': 'graphs/P2PAgent_10A_0E_50B_1V_complete(directed)_N10_NB9_TV-1_04-01-2022_17_18',
            'Sparse (directed)': 'graphs/P2PAgent_10A_0E_50B_1V_sparse(directed)_N10_NB3_TV-1_04-01-2022_17_19',
        }, n_rows=2
    )


def plot_experiment_1():
    side_by_side(
        {
            'Ring undirected (non-BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',
                    ],
                    'SGP': [
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',
                    ],
                    ALG_NAME: [
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_03_22',
                    ],
                },
            },
            'Ring undirected (BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_01_09',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_21_25',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_05',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                    ],
                    'SGP': [
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_27',
                    ],
                    ALG_NAME: [
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_42',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                    ],
                },
            },
            'Ring directed (non-BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',
                    ],
                    'SGP': [
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',
                    ],
                    ALG_NAME: [
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_20_21',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_19_08',
                    ],
                },
            },
            'Ring directed (BN)': {
                'x_axis': 'epoch',
                'viz': {
                    'D$^2$': [
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_14_03',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_48',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_16_53',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_09',
                    ],
                    'SGP': [
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_02',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_39',
                    ],
                    ALG_NAME: [
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_54',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_14',
                    ],
                },
            },
        }, fig_size=(12/2, 12 / 3.75*2), n_rows=2)


def plot_experiment_1_graph():
    plot_graph(
        {
            'Undirected ring': 'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
            'Directed ring': 'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
        }
    )


def plot_experiment_2():
    side_by_side({
        'Fixed sparse (undirected)': {
            'x_axis': 'epoch',
            'viz': {
                'D$^2$': [
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_53',
                ],
                'GoSGD': [
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_00_26',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_20_50',
                ],
                'SGP': [
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_47',
                ],
                ALG_NAME: [
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_12_58',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_17_52',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                ]
            }
        },
        'Fixed sparse (directed)': {
            'x_axis': 'epoch',
            'viz': {
                'D$^2$': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ],
                ALG_NAME: [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                ]
            }
        },
        'Varying sparse (undirected)': {
            'x_axis': 'epoch',
            'viz': {
                'D$^2$': [
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_20_50',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_00',
                ],
                'GoSGD': [
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_19_59',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_23_44',
                ],
                'SGP': [
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_21_12',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_15',
                ],
                ALG_NAME: [
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_12_09',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_16_55',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                ]
            }
        },
        'Varying sparse (directed)': {
            'x_axis': 'epoch',
            'viz': {
                'D$^2$': [
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_30',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_02_32',

                ],
                'GoSGD': [
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_18_05',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_22_10',
                ],
                'SGP': [
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_27',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_03_12',
                ],
                ALG_NAME: [
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_10_28',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_13_51',
                ],
                'FL': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                ]
            }
        },
    }, fig_size=(12 / 2, 12 / 3.75 * 2), n_rows=2)


def exp_1_avg_msg_acc():
    viz = {
            'D$^2$': [
                # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',

                'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_01_09',
                'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_21_25',

                # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',

                'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
                'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
                'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
                'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_14_03',
                'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_48',

            ],
            'GoSGD': [
                # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',

                'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
                'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
                'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
                'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_05',
                'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',

                # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',

                'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
                'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
                'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
                'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_16_53',
                'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_09',
            ],
            'SGP': [
                # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',

                'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_27',

                # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',

                'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
                'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
                'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_02',
                'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_39',
            ],
            ALG_NAME: [
                # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
                # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
                # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
                # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                # 'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_03_22',

                'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
                'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
                'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
                'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_42',
                'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',

                # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
                # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
                # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
                # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_20_21',
                # 'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_19_08',

                'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
                'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
                'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
                'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_54',
                'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_14',
            ],
        }

    print("Average number of communications per each epoch")
    for k, v in viz.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))


def exp_2_avg_msg_acc():
    viz = {
        ALG_NAME: [
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_12_58',
            'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_17_52',

            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
            'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',

            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_12_09',
            'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_16_55',

            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_10_28',
            'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_13_51',
        ],
        'D$^2$': [
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
            'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_53',

            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
            'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',

            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_20_50',
            'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_00',

            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_30',
            'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_02_32',

        ],
        'GoSGD': [
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_00_26',
            'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_20_50',

            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
            'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',

            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_19_59',
            'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_23_44',

            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_18_05',
            'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_22_10',

        ],
        'SGP': [
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
            'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_47',

            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
            'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',

            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_21_12',
            'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_15',

            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_27',
            'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_03_12',

        ],
        'FL': [
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
            'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
        ]
    }

    print("Average number of communications per each epoch and avg top accuracy")
    for k, v in viz.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))


def plot_experiment_2_graph():
    plot_graph(
        {
            'Undirected sparse': 'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
            'Directed sparse': 'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
        }, node_size=100,
    )


def plot_experiment_3():
    side_by_side({
        '': {
            'x_axis': 'epoch',
            'viz': {
                ALG_NAME + ' (100)': [
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_05',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_22',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_45',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_17_11',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_18_27',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_07_45',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_13_52',
                ],
                ALG_NAME + ' (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_29-12-2021_09_51',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_26-12-2021_05_47',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_23_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_10_46',
                ],
                'FL (100)': [
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                ],
                'FL (500)': [
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_12_24',
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_17_21',
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_22_20',
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_07_41',
                    'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_13_06',
                ],
                'FL (1000)': [
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_09-12-2021_18_57',
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_10-12-2021_05_17',
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_10-12-2021_16_09',
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_16-03-2022_00_54',
                    'experiment_3/fl_1000C_10TR_2V(0_005S-0_005C)_100E_num_examples_16-03-2022_13_36',
                ]
            }
        },
        ' ': {
            'x_axis': 'acomms',
            'viz': {
                ALG_NAME + ' (100)': [
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_05',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_22',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_45',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_17_11',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_18_27',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_07_45',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_13_52',
                ],
                ALG_NAME + ' (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_29-12-2021_09_51',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_26-12-2021_05_47',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_23_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_10_46',
                ]
            }
        }
    }, fig_size=(7, 7 / 2))


def exp_bn_ft():
    side_by_side({
        'Sparse (directed)': {
            'x_axis': 'epoch',
            'viz': {
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_02_41',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_01',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_24',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_22',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_31',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_12_07',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_18_02',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_19_20',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_21_05',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_20_24',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_53',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_10_56',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_04_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_23',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_40',
                ],
                ALG_NAME + ' (300)': [
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_01_48',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_04_53',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_07_17',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_48',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_00',
                ],
            }
        },
        'Sparse (undirected)': {
            'x_axis': 'epoch',
            'viz': {
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_16_50',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_07',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_33',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_16',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_20',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_07_21',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_06_24',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_04_12',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_20_34',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_21_18',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_54',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_22-01-2022_00_46',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_11',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_09',
                ],
                ALG_NAME + ' (300)': [
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_35',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_19_17',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_43',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_01_25',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_02_49',
                ],
            }
        },
    }, fig_size=(7, 7/2))


def bn_ft_significance():
    viz = {
            ALG_NAME + ' (100 - no BN FT) - directed': [
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_02_41',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_01',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_24',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_22',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_31',
            ],
            ALG_NAME + ' (300 - no BN FT) - directed': [
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_12_07',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_18_02',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_19_20',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_21_05',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_20_24',
            ],
            ALG_NAME + ' (100 - no BN FT) - undirected': [
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_16_50',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_07',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_33',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_16',
                'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_20',
            ],
            ALG_NAME + ' (300 - no BN FT) - undirected': [
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_07_21',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_06_24',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_04_12',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_20_34',
                'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_21_18',

            ],
            ALG_NAME + ' (100) - directed': [
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_53',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_10_56',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_04_15',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_23',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_40',
            ],
            ALG_NAME + ' (300) - directed': [
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_01_48',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_04_53',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_07_17',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_48',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_00',
            ],
            ALG_NAME + ' (100) - undirected': [
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_15',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_54',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_22-01-2022_00_46',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_11',
                'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_09',
            ],
            ALG_NAME + ' (300) - undirected': [
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_35',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_19_17',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_43',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_01_25',
                'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_02_49',
            ],
        }

    accs = []
    for k, v in viz.items():
        _, acc, _ = parse_timeline(None, v)
        accs.append(acc)

    half_len = int(len(accs)/2)
    names = list(viz.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i+half_len], mannwhitneyu(accs[i][20:81], accs[i+half_len][20:]))


def independent_acc():
    viz = {
            'agents': [
                'experiment_2/indp/P2PAgent_100A_100E_50B_4V_21-02-2022_21_20',
                'experiment_2/indp/P2PAgent_100A_100E_50B_4V_22-02-2022_00_01',
                'experiment_2/indp/P2PAgent_100A_100E_50B_4V_22-02-2022_03_13',
                'experiment_2/indp/P2PAgent_100A_100E_50B_4V_15-03-2022_02_21',
                'experiment_2/indp/P2PAgent_100A_100E_50B_4V_15-03-2022_03_14',
            ],
            'single': [
                'experiment_2/indp/Single_100A_100E_50B_4V_22-02-2022_11_44',
                'experiment_2/indp/Single_100A_100E_50B_4V_22-02-2022_12_26',
                'experiment_2/indp/Single_100A_100E_50B_4V_22-02-2022_13_17',
                'experiment_2/indp/Single_100A_100E_50B_4V_15-03-2022_00_42',
                'experiment_2/indp/Single_100A_100E_50B_4V_14-03-2022_23_38',
            ]
        }

    for k, v in viz.items():
        max_acc = []
        for filename in v:
            data = read_json(filename)['agents']
            if isinstance(data, list):
                data = np.array(data)
                m_ac = [max(data[:, i]) for i in range(data.shape[1])]
            else:
                m_ac = [max(data[key]['test_model-accuracy_no_oov']) for key in data.keys()]
            max_acc.append(m_ac)
        print(k, "{:.3%}".format(np.average(max_acc)))


if __name__ == '__main__':
    plot_experiment_2_graph()
