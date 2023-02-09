from plot.visualize import side_by_side, show, plot_graph, resolve_timeline, parse_timeline, read_json
from scipy.stats import mannwhitneyu, ttest_ind
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
    colors = ['r', 'g', 'b', 'indigo']
    side_by_side(
        {
            'Ring undirected (non-BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_00',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_18_49',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                        'experiment_1/undirected/nonbn/P2PAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_03_22',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                        'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',
                    ],
                    'D$^2$': [
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                        'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',
                    ],
                    'SGP': [
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                        'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',
                    ],
                },
            },
            'Ring undirected (BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_54',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_12_08',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_42',
                        'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_02_16',
                    ],
                    'GoSGD': [
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_12',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_10_14',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_20-12-2021_12_40',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_05',
                        'experiment_1/undirected/bn/GoSGDAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                    ],
                    'D$^2$': [
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_49',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_18_45',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_01',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_01_09',
                        'experiment_1/undirected/bn/D2Agent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_21_25',
                    ],
                    'SGP': [
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_17_55',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_28-11-2021_22_21',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_11-03-2022_20_48',
                        'experiment_1/undirected/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_12-03-2022_00_27',
                    ],
                },
            },
            'Ring directed (non-BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_17_58',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_15',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_18_48',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_20_21',
                        'experiment_1/directed/nonbn/P2PAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_11-03-2022_19_08',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                        'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',
                    ],
                    'D$^2$': [
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                        'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',
                    ],
                    'SGP': [
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                        'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',
                    ],
                },
            },
            'Ring directed (BN)': {
                'x_axis': 'epoch',
                'colors': colors,
                'viz': {
                    ALG_NAME: [
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_07',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_22_32',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_54',
                        'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_14',
                    ],
                    'GoSGD': [
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_20',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_58',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_21-12-2021_05_32',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_16_53',
                        'experiment_1/directed/bn/GoSGDAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_09',
                    ],
                    'D$^2$': [
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_50',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_19',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_18_31',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_14_03',
                        'experiment_1/directed/bn/D2Agent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_48',
                    ],
                    'SGP': [
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_15_46',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_17-12-2021_16_36',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_13_02',
                        'experiment_1/directed/bn/SGPPushSumAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_12-03-2022_17_39',
                    ],
                },
            },
        }, fig_size=(12 / 2, 12 / 3.75 * 2), n_rows=2)


def plot_experiment_1_graph():
    plot_graph(
        {
            'Undirected ring': 'experiment_1/undirected/bn/P2PAgent_100A_100E_50B_4V_ring(undirected)_N100_NB2_TV-1_16-12-2021_11_36',
            'Directed ring': 'experiment_1/directed/bn/P2PAgent_100A_100E_50B_4V_ring(directed)_N100_NB1_TV-1_16-12-2021_21_48',
        }
    )


def plot_experiment_2():
    colors = ['r', 'g', 'b', 'indigo']
    side_by_side({
        'Fixed sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_10_08',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_12_46',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_24-11-2021_15_53',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_12_58',
                    'experiment_2/fixed/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_17_52',
                ],
                'GoSGD': [
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_09_10',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_10_03',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_17',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_00_26',
                    'experiment_2/fixed/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_20_50',
                ],
                'D$^2$': [
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_37',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_02',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_15_59',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                    'experiment_2/fixed/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_53',
                ],
                'SGP': [
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_12_19',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_13_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_26-11-2021_17_00',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_13-03-2022_01_55',
                    'experiment_2/fixed/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_12-03-2022_21_47',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                #  ]
            }
        },
        'Fixed sparse (directed)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'D$^2$': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                # ]
            }
        },
        'Varying sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_17_01',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_20_17',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_02-12-2021_00_57',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_12_09',
                    'experiment_2/varying/undirected/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_16_55',
                ],
                'GoSGD': [
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_45',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_10_48',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_50',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_19_59',
                    'experiment_2/varying/undirected/GoSGDAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_23_44',
                ],
                'D$^2$': [
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_07',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_14_08',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_18',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_20_50',
                    'experiment_2/varying/undirected/D2Agent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_00',
                ],
                'SGP': [
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_12_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_13_58',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_01-12-2021_16_41',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_13-03-2022_21_12',
                    'experiment_2/varying/undirected/SGPPushSumAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV5_14-03-2022_01_15',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                #  ]
            }
        },
        'Varying sparse (directed)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_23_24',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_18_46',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_14_59',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_10_28',
                    'experiment_2/varying/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_13_51',
                ],
                'GoSGD': [
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_08_44',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_10',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_36',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_18_05',
                    'experiment_2/varying/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_22_10',
                ],
                'D$^2$': [
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_09_44',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_02-12-2021_10_13',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_14',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_30',
                    'experiment_2/varying/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_02_32',

                ],
                'SGP': [
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_04_28',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_03_12',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_03-12-2021_02_29',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_14-03-2022_19_27',
                    'experiment_2/varying/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV5_15-03-2022_03_12',
                ],
                # 'FL': [
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_00_56',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_02_09',
                #    'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_27-11-2021_03_34',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_14-03-2022_23_55',
                #     'experiment_2/fl_100C_10TR_2V(0_005S-0_005C)_100E_num_examples_15-03-2022_01_05',
                #  ]
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
    colors = ['red', 'coral', 'orange', 'blue', 'cornflowerblue', 'g', 'purple', 'limegreen', 'violet']
    viz = {
        '': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME + ' (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_29-12-2021_09_51',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_26-12-2021_05_47',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_23_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_10_46',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_07_45',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_13_52',
                ],
                ALG_NAME + ' (100)': [
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_05',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_22',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_45',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_17_11',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_18_27',
                ],
                'GoSGD (500)': [
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_18-06-2022_03_34',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_20-06-2022_01_42',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_13-06-2022_15_01',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-06-2022_05_17',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_21-06-2022_10_02',
                ],
                'GoSGD (100)': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'D$^2$ (500)': [
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_17-06-2022_23_24',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_25-06-2022_18_34',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_23-06-2022_03_19',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_03-07-2022_03_36',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_13_45',
                ],
                'SGP (500)': [
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_25-06-2022_23_49',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_14_17',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_16_30',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_05-07-2022_05_11',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_04-07-2022_18_36',
                ],
                'D$^2$ (100)': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'SGP (100)': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ]
            }
        },
    }

    side_by_side(viz, fig_size=(5, 5))
    viz = viz['']['viz']
    # 5 points
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for avg_acc in [parse_timeline(None, val)[1] for val in v]:
            avg_acc = np.array(avg_acc)[20:]
            avg_max = np.max(avg_acc)
            accs[k].append(avg_max)
        accs[k] = np.array(accs[k]).flatten()

    avg_accs = {}
    for k, v in viz.items():
        _, acc, _ = parse_timeline(None, v)
        avg_accs[k] = acc

    for name in [ALG_NAME, 'GoSGD', 'D$^2$', 'SGP']:
        print(name + ' (100 <-> 500) Max100:', max(avg_accs[name + ' (100)']), 'Max500:',
              max(avg_accs[name + ' (500)']), 'TTEST:', ttest_ind(accs[name + ' (100)'], accs[name + ' (500)']))


def exp_bn_ft():
    side_by_side({
        'Sparse (directed)': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (300)': [
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_01_48',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_04_53',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_20-01-2022_07_17',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_48',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_04_00',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_53',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_10_56',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_04_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_23',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_40',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_12_07',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_18_02',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_19-01-2022_19_20',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_21_05',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(directed)_N300_NB3_TV-1_10-03-2022_20_24',
                ],
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_02_41',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_01',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_24',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_22',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_31',
                ],
            }
        },
        'Sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (300)': [
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_35',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_19_17',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_20_43',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_01_25',
                    'exp_es/es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_02_49',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_54',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_22-01-2022_00_46',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_11',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_09',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_07_21',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_06_24',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_22-01-2022_04_12',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_20_34',
                    'exp_es/no_es/P2PAgent_300A_100E_50B_4V_sparse(undirected)_N300_NB3_TV-1_10-03-2022_21_18',
                ],
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_16_50',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_07',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_21-01-2022_17_33',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_16_16',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(undirected)_N100_NB3_TV-1_09-03-2022_17_20',
                ],
            }
        },
    }, fig_size=(7, 7 / 2))


def exp_bn_ft_so():
    viz = {
        'Sparse (directed)': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (300)': [
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_17-09-2022_04_28',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_20-09-2022_07_23',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_20-09-2022_23_39',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_24-10-2022_11_07',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_26-10-2022_07_15',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_15-09-2022_04_28',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_10_45',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_17_26',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_07-10-2022_23_19',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_08-10-2022_20_35',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_17-09-2022_05_41',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_23-09-2022_00_19',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_23-09-2022_19_51',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_14-10-2022_22_29',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(directed-3)_19-10-2022_04_40',
                ],
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_15-09-2022_03_55',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_00_42',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_13_32',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_10-10-2022_20_12',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_19_14',
                ],
            }
        },
        'Sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (300)': [
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_17-09-2022_05_01',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_21-09-2022_20_56',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_22-09-2022_10_38',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_23-10-2022_23_00',
                    'exp_es/es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_22-10-2022_07_53',
                ],
                ALG_NAME + ' (100)': [
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_15-09-2022_04_25',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_19-09-2022_10_34',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_19-09-2022_17_23',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_09-10-2022_11_29',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_10-10-2022_02_03',
                ],
                ALG_NAME + ' (300 - no BN FT)': [
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_17-09-2022_05_39',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_29-09-2022_21_24',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_01-10-2022_11_42',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_20-10-2022_19_55',
                    'exp_es/no_es/so/P2PAgent_300A_100E_50B_sparse(undirected-3)_22-10-2022_16_10',
                ],
                ALG_NAME + ' (100 - no BN FT)': [
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_15-09-2022_03_50',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_19-09-2022_01_00',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_19-09-2022_13_16',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_12-10-2022_10_11',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_12-10-2022_23_59',
                ],
            }
        },
    }
    side_by_side(viz, fig_size=(7, 7 / 2))
    orig_viz = viz
    for title in ['Sparse (directed)', 'Sparse (undirected)']:
        print(title)
        viz = orig_viz[title]['viz']
        accs = {}
        for k, v in viz.items():
            accs[k] = []
            for avg_acc in [parse_timeline(None, val)[1] for val in v]:
                avg_acc = np.array(avg_acc)[20:]
                avg_max = np.max(avg_acc)
                accs[k].append(avg_max)
            accs[k] = np.array(accs[k]).flatten()

        half_len = int(len(accs) / 2)
        names = list(accs.keys())
        for i in range(half_len):
            # print(names[i], "<->", names[i + half_len], ttest_ind(accs[names[i]], accs[names[i + half_len]]))
            print(names[i + half_len], "<->", names[i], ttest_ind(accs[names[i + half_len]], accs[names[i]]))
        print("Max accs")
        for k, v in viz.items():
            _, acc, _ = parse_timeline(None, v)
            print(k, max(acc))


def plot_so_experiment_1():
    colors = ['r', 'g', 'b', 'indigo']
    viz = {
        'Ring undirected (non-BN)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_1/undirected/nonbn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_15-09-2022_04_20',
                    'experiment_1/undirected/nonbn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_20-09-2022_05_54',
                    'experiment_1/undirected/nonbn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_20-09-2022_18_50',
                    'experiment_1/undirected/nonbn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_07-10-2022_20_28',
                    'experiment_1/undirected/nonbn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_08-10-2022_10_52',
                ],
                'GoSGD': [
                    'experiment_1/undirected/nonbn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_16-09-2022_03_20',
                    'experiment_1/undirected/nonbn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_20-09-2022_18_21',
                    'experiment_1/undirected/nonbn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_22-09-2022_02_47',
                    'experiment_1/undirected/nonbn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_08-10-2022_15_26',
                    'experiment_1/undirected/nonbn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_10-10-2022_01_48',
                ],
                'D$^2$': [
                    'experiment_1/undirected/nonbn/so/D2Agent_100A_100E_50B_ring(undirected-2)_17-09-2022_16_46',
                    'experiment_1/undirected/nonbn/so/D2Agent_100A_100E_50B_ring(undirected-2)_20-09-2022_21_07',
                    'experiment_1/undirected/nonbn/so/D2Agent_100A_100E_50B_ring(undirected-2)_22-09-2022_05_37',
                    'experiment_1/undirected/nonbn/so/D2Agent_100A_100E_50B_ring(undirected-2)_09-10-2022_16_39',
                    'experiment_1/undirected/nonbn/so/D2Agent_100A_100E_50B_ring(undirected-2)_11-10-2022_18_34',
                ],
                'SGP': [
                    'experiment_1/undirected/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_16-09-2022_10_16',
                    'experiment_1/undirected/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_21-09-2022_01_53',
                    'experiment_1/undirected/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_21-09-2022_23_13',
                    'experiment_1/undirected/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_09-10-2022_13_34',
                    'experiment_1/undirected/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_11-10-2022_16_43',
                ],
            },
        },
        'Ring undirected (BN)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_1/undirected/bn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_15-09-2022_04_38',
                    'experiment_1/undirected/bn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_21-09-2022_06_38',
                    'experiment_1/undirected/bn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_21-09-2022_20_08',
                    'experiment_1/undirected/bn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_10-10-2022_21_25',
                    'experiment_1/undirected/bn/so/P2PAgent_100A_100E_50B_ring(undirected-2)_11-10-2022_20_09',
                ],
                'GoSGD': [
                    'experiment_1/undirected/bn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_16-09-2022_04_20',
                    'experiment_1/undirected/bn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_23-09-2022_06_20',
                    'experiment_1/undirected/bn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_24-09-2022_12_52',
                    'experiment_1/undirected/bn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_15-10-2022_13_34',
                    'experiment_1/undirected/bn/so/GoSGDAgent_100A_100E_50B_ring(undirected-2)_16-10-2022_18_30',
                ],
                'D$^2$': [
                    'experiment_1/undirected/bn/so/D2Agent_100A_100E_50B_ring(undirected-2)_17-09-2022_19_10',
                    'experiment_1/undirected/bn/so/D2Agent_100A_100E_50B_ring(undirected-2)_23-09-2022_07_04',
                    'experiment_1/undirected/bn/so/D2Agent_100A_100E_50B_ring(undirected-2)_24-09-2022_13_12',
                    'experiment_1/undirected/bn/so/D2Agent_100A_100E_50B_ring(undirected-2)_20-10-2022_06_55',
                    'experiment_1/undirected/bn/so/D2Agent_100A_100E_50B_ring(undirected-2)_21-10-2022_14_15',
                ],
                'SGP': [
                    'experiment_1/undirected/bn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_16-09-2022_12_16',
                    'experiment_1/undirected/bn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_23-09-2022_21_16',
                    'experiment_1/undirected/bn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-2)_24-09-2022_15_07',
                    'experiment_1/undirected/bn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-1)_26-10-2022_02_36',
                    'experiment_1/undirected/bn/so/SGPPushSumAgent_100A_100E_50B_ring(undirected-1)_26-10-2022_00_05',
                ],
            },
        },
        'Ring directed (non-BN)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_1/directed/nonbn/so/P2PAgent_100A_100E_50B_ring(directed-1)_15-09-2022_04_07',
                    'experiment_1/directed/nonbn/so/P2PAgent_100A_100E_50B_ring(directed-1)_23-09-2022_05_40',
                    'experiment_1/directed/nonbn/so/P2PAgent_100A_100E_50B_ring(directed-1)_23-09-2022_18_04',
                    'experiment_1/directed/nonbn/so/P2PAgent_100A_100E_50B_ring(directed-1)_09-10-2022_09_24',
                    'experiment_1/directed/nonbn/so/P2PAgent_100A_100E_50B_ring(directed-1)_09-10-2022_23_53',
                ],
                'GoSGD': [
                    'experiment_1/directed/nonbn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_16-09-2022_02_51',
                    'experiment_1/directed/nonbn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_29-09-2022_04_34',
                    'experiment_1/directed/nonbn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_29-09-2022_10_26',
                    'experiment_1/directed/nonbn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_11-10-2022_14_12',
                    'experiment_1/directed/nonbn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_12-10-2022_20_11',
                ],
                'D$^2$': [
                    'experiment_1/directed/nonbn/so/D2Agent_100A_100E_50B_ring(directed-1)_17-09-2022_15_55',
                    'experiment_1/directed/nonbn/so/D2Agent_100A_100E_50B_ring(directed-1)_25-09-2022_17_25',
                    'experiment_1/directed/nonbn/so/D2Agent_100A_100E_50B_ring(directed-1)_25-09-2022_01_11',
                    'experiment_1/directed/nonbn/so/D2Agent_100A_100E_50B_ring(directed-1)_15-10-2022_06_16',
                    'experiment_1/directed/nonbn/so/D2Agent_100A_100E_50B_ring(directed-1)_18-10-2022_16_35',
                ],
                'SGP': [
                    'experiment_1/directed/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_16-09-2022_08_51',
                    'experiment_1/directed/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_22-09-2022_12_14',
                    'experiment_1/directed/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_23-09-2022_05_01',
                    'experiment_1/directed/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_14-10-2022_16_15',
                    'experiment_1/directed/nonbn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_18-10-2022_03_58',
                ],
            },
        },
        'Ring directed (BN)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_1/directed/bn/so/P2PAgent_100A_100E_50B_ring(directed-1)_15-09-2022_04_40',
                    'experiment_1/directed/bn/so/P2PAgent_100A_100E_50B_ring(directed-1)_22-09-2022_06_46',
                    'experiment_1/directed/bn/so/P2PAgent_100A_100E_50B_ring(directed-1)_22-09-2022_18_54',
                    'experiment_1/directed/bn/so/P2PAgent_100A_100E_50B_ring(directed-1)_12-10-2022_11_12',
                    'experiment_1/directed/bn/so/P2PAgent_100A_100E_50B_ring(directed-1)_13-10-2022_00_55',
                ],
                'GoSGD': [
                    'experiment_1/directed/bn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_16-09-2022_03_46',
                    'experiment_1/directed/bn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_25-09-2022_19_09',
                    'experiment_1/directed/bn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_27-09-2022_00_36',
                    'experiment_1/directed/bn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_18-10-2022_12_16',
                    'experiment_1/directed/bn/so/GoSGDAgent_100A_100E_50B_ring(directed-1)_19-10-2022_19_44',
                ],
                'D$^2$': [
                    'experiment_1/directed/bn/so/D2Agent_100A_100E_50B_ring(directed-1)_17-09-2022_18_51',
                    'experiment_1/directed/bn/so/D2Agent_100A_100E_50B_ring(directed-1)_26-09-2022_06_09',
                    'experiment_1/directed/bn/so/D2Agent_100A_100E_50B_ring(directed-1)_28-09-2022_11_44',
                    'experiment_1/directed/bn/so/D2Agent_100A_100E_50B_ring(directed-1)_23-10-2022_00_22',
                    'experiment_1/directed/bn/so/D2Agent_100A_100E_50B_ring(directed-1)_24-10-2022_15_19',
                ],
                'SGP': [
                    'experiment_1/directed/bn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_16-09-2022_11_29',
                    'experiment_1/directed/bn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_25-09-2022_09_30',
                    'experiment_1/directed/bn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_26-09-2022_05_38',
                    'experiment_1/directed/bn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_25-10-2022_21_11',
                    'experiment_1/directed/bn/so/SGPPushSumAgent_100A_100E_50B_ring(directed-1)_25-10-2022_18_28',
                ],
            },
        },
    }
    side_by_side(viz, fig_size=(12 / 2, 12 / 3.75 * 2), n_rows=2)

    print("Statistics")
    data = {}
    for gk, g in viz.items():
        if 'non-BN' in gk:
            # print("Skipping", gk)
            continue
        for k, v in g['viz'].items():
            if k in data:
                data[k].extend(v)
            else:
                data[k] = v
    for k, v in data.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))


def plot_so_experiment_2():
    colors = ['r', 'g', 'b', 'indigo']
    viz = {
        'Fixed sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_18-09-2022_22_33',
                    'experiment_2/fixed/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_26-09-2022_18_41',
                    'experiment_2/fixed/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_27-09-2022_09_43',
                    'experiment_2/fixed/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_09-10-2022_12_08',
                    'experiment_2/fixed/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_10-10-2022_02_11',
                ],
                'GoSGD': [
                    'experiment_2/fixed/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_17-09-2022_14_57',
                    'experiment_2/fixed/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_28-09-2022_11_52',
                    'experiment_2/fixed/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_30-09-2022_12_38',
                    'experiment_2/fixed/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_08-10-2022_17_20',
                    'experiment_2/fixed/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_10-10-2022_03_40',
                ],
                'D$^2$': [
                    'experiment_2/fixed/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_17-09-2022_19_28',
                    'experiment_2/fixed/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_29-09-2022_13_05',
                    'experiment_2/fixed/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_30-09-2022_18_02',
                    'experiment_2/fixed/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_09-10-2022_19_29',
                    'experiment_2/fixed/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_11-10-2022_20_13',
                ],
                'SGP': [
                    'experiment_2/fixed/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-09-2022_17_19',
                    'experiment_2/fixed/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_26-09-2022_22_54',
                    'experiment_2/fixed/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_28-09-2022_00_38',
                    'experiment_2/fixed/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_08-10-2022_21_56',
                    'experiment_2/fixed/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_10-10-2022_07_28',
                ],
            }
        },
        'Fixed sparse (directed)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_22_25',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_27-09-2022_20_15',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_09_10',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_07-10-2022_22_56',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_08-10-2022_14_22',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-09-2022_14_16',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_01-10-2022_13_02',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_02-10-2022_19_35',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_16_31',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_12-10-2022_21_49',
                ],
                'D$^2$': [
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_17-09-2022_19_33',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_01-10-2022_19_06',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_03-10-2022_02_15',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_15-10-2022_10_13',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_18-10-2022_20_38',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_17_26',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_20_12',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_30-09-2022_01_15',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_20_52',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_13-10-2022_17_08',
                ],
            }
        },
        'Varying sparse (undirected)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/varying/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_17-09-2022_21_45',
                    'experiment_2/varying/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_29-09-2022_16_46',
                    'experiment_2/varying/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_29-09-2022_05_35',
                    'experiment_2/varying/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_12-10-2022_11_14',
                    'experiment_2/varying/undirected/so/P2PAgent_100A_100E_50B_sparse(undirected-3)_13-10-2022_00_59',
                ],
                'GoSGD': [
                    'experiment_2/varying/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-09-2022_15_36',
                    'experiment_2/varying/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_03-10-2022_22_02',
                    'experiment_2/varying/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_05-10-2022_03_51',
                    'experiment_2/varying/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-10-2022_12_17',
                    'experiment_2/varying/undirected/so/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_19-10-2022_20_03',
                ],
                'D$^2$': [
                    'experiment_2/varying/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_18-09-2022_17_48',
                    'experiment_2/varying/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_04-10-2022_05_46',
                    'experiment_2/varying/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_03-10-2022_05_24',
                    'experiment_2/varying/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_20-10-2022_07_51',
                    'experiment_2/varying/undirected/so/D2Agent_100A_100E_50B_sparse(undirected-3)_21-10-2022_14_39',
                ],
                'SGP': [
                    'experiment_2/varying/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-09-2022_18_11',
                    'experiment_2/varying/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_30-09-2022_20_26',
                    'experiment_2/varying/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_01-10-2022_16_29',
                    'experiment_2/varying/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-10-2022_15_30',
                    'experiment_2/varying/undirected/so/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_19-10-2022_23_32',
                ],
            }
        },
        'Varying sparse (directed)': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/varying/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_17-09-2022_21_43',
                    'experiment_2/varying/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_16_27',
                    'experiment_2/varying/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_01_19',
                    'experiment_2/varying/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_10-10-2022_21_43',
                    'experiment_2/varying/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_20_15',
                ],
                'GoSGD': [
                    'experiment_2/varying/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_15_46',
                    'experiment_2/varying/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_05-10-2022_03_55',
                    'experiment_2/varying/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_09_02',
                    'experiment_2/varying/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_15-10-2022_14_11',
                    'experiment_2/varying/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_16-10-2022_18_39',
                ],
                'D$^2$': [
                    'experiment_2/varying/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_18-09-2022_17_43',
                    'experiment_2/varying/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_01-10-2022_23_03',
                    'experiment_2/varying/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_30-09-2022_22_36',
                    'experiment_2/varying/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_23-10-2022_01_54',
                    'experiment_2/varying/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_24-10-2022_16_17',

                ],
                'SGP': [
                    'experiment_2/varying/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_18_13',
                    'experiment_2/varying/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_02-10-2022_07_04',
                    'experiment_2/varying/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_03-10-2022_01_27',
                    'experiment_2/varying/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_15-10-2022_18_14',
                    'experiment_2/varying/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-10-2022_00_54',
                ],
            }
        },
    }
    side_by_side(viz, fig_size=(12 / 2, 12 / 3.75 * 2), n_rows=2)

    print("Statistics")
    data = {}
    for _, g in viz.items():
        for k, v in g['viz'].items():
            if k in data:
                data[k].extend(v)
            else:
                data[k] = v
    for k, v in data.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))


def plot_so_experiment_3():
    colors = ['red', 'coral', 'orange', 'blue', 'cornflowerblue', 'g', 'purple', 'limegreen', 'violet']
    viz = {
        '': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME + ' (1000)': [
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_09-10-2022_21_47',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_10-10-2022_21_28',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_13-10-2022_07_30',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_13-10-2022_12_59',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_16-10-2022_21_09',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_20-09-2022_17_44',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_22-09-2022_03_17',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_23-09-2022_12_57',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_14-10-2022_11_51',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_15-10-2022_12_38',
                ],
                ALG_NAME + ' (100)': [
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_10_22',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_15_16',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_21_47',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_25-10-2022_14_18',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_25-10-2022_13_28',
                ],
                'GoSGD (500)': [
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_07-10-2022_22_54',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_13-10-2022_00_21',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_08-10-2022_05_42',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_18-10-2022_20_07',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_23-10-2022_07_48',
                ],
                'GoSGD (100)': [
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-09-2022_14_16',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_01-10-2022_13_02',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_02-10-2022_19_35',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_16_31',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_12-10-2022_21_49',
                ],
                'D$^2$ (500)': [
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_28-09-2022_09_44',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_06-10-2022_18_28',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_23-10-2022_14_45',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_30-10-2022_18_19',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_31-10-2022_01_35',
                ],
                'SGP (500)': [
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_27-09-2022_23_18',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_06-10-2022_17_37',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_25-10-2022_10_19',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_30-10-2022_19_01',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_01-11-2022_19_41',
                ],
                'D$^2$ (100)': [
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_17-09-2022_19_33',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_01-10-2022_19_06',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_03-10-2022_02_15',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_15-10-2022_10_13',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_18-10-2022_20_38',
                ],
                'SGP (100)': [
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_17_26',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_20_12',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_30-09-2022_01_15',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_20_52',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_13-10-2022_17_08',
                ],
            }
        },
    }

    side_by_side(viz, fig_size=(5, 5))
    viz = viz['']['viz']
    # 5 points
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for avg_acc in [parse_timeline(None, val)[1] for val in v]:
            avg_acc = np.array(avg_acc)[20:]
            avg_max = np.max(avg_acc)
            accs[k].append(avg_max)
        accs[k] = np.array(accs[k]).flatten()

    avg_accs = {}
    for k, v in viz.items():
        _, acc, _ = parse_timeline(None, v)
        avg_accs[k] = acc

    for name in [ALG_NAME, 'GoSGD', 'D$^2$', 'SGP']:
        print(name + ' (100 <-> 500) Max100:', max(avg_accs[name + ' (100)']), 'Max500:',
              max(avg_accs[name + ' (500)']), 'TTEST:', ttest_ind(accs[name + ' (100)'], accs[name + ' (500)']))


def plot_reddit_so_experiment_3():
    colors = ['red', 'coral', 'orange', 'blue', 'cornflowerblue', 'g', 'purple', 'limegreen', 'violet']
    viz = {
        'Reddit': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME + ' (1000)': [
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_29-12-2021_09_51',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_26-12-2021_05_47',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_23_30',
                    'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_17-03-2022_10_46',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_16_01',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_16-12-2021_11_21',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_07_45',
                    'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_15-03-2022_13_52',
                ],
                ALG_NAME + ' (100)': [
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_05',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_22',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_45',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_17_11',
                    'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_18_27',
                ],
                'GoSGD (500)': [
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_18-06-2022_03_34',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_20-06-2022_01_42',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_13-06-2022_15_01',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-06-2022_05_17',
                    'experiment_3/GoSGDAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_21-06-2022_10_02',
                ],
                'GoSGD (100)': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'D$^2$ (500)': [
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_17-06-2022_23_24',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_25-06-2022_18_34',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_23-06-2022_03_19',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_03-07-2022_03_36',
                    'experiment_3/D2Agent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_13_45',
                ],
                'SGP (500)': [
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_25-06-2022_23_49',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_14_17',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_30-06-2022_16_30',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_05-07-2022_05_11',
                    'experiment_3/SGPPushSumAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_04-07-2022_18_36',
                ],
                'D$^2$ (100)': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'SGP (100)': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ]
            }
        },
        'StackOverflow': {
            'x_axis': 'epoch',
            'colors': colors,
            'viz': {
                ALG_NAME + ' (1000)': [
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_09-10-2022_21_47',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_10-10-2022_21_28',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_13-10-2022_07_30',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_13-10-2022_12_59',
                    'experiment_3/so/P2PAgent_1000A_100E_50B_sparse(directed-3)_16-10-2022_21_09',
                ],
                ALG_NAME + ' (500)': [
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_20-09-2022_17_44',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_22-09-2022_03_17',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_23-09-2022_12_57',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_14-10-2022_11_51',
                    'experiment_3/so/P2PAgent_500A_100E_50B_sparse(directed-3)_15-10-2022_12_38',
                ],
                ALG_NAME + ' (100)': [
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_10_22',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_15_16',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_04-10-2022_21_47',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_25-10-2022_14_18',
                    'experiment_3/so/P2PAgent_100A_100E_50B_sparse(directed-3)_25-10-2022_13_28',
                ],
                'GoSGD (500)': [
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_07-10-2022_22_54',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_13-10-2022_00_21',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_08-10-2022_05_42',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_18-10-2022_20_07',
                    'experiment_3/so/GoSGDAgent_500A_100E_50B_sparse(directed-3)_23-10-2022_07_48',
                ],
                'GoSGD (100)': [
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-09-2022_14_16',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_01-10-2022_13_02',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_02-10-2022_19_35',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_16_31',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_12-10-2022_21_49',
                ],
                'D$^2$ (500)': [
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_28-09-2022_09_44',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_06-10-2022_18_28',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_23-10-2022_14_45',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_30-10-2022_18_19',
                    'experiment_3/so/D2Agent_500A_100E_50B_sparse(directed-3)_31-10-2022_01_35',
                ],
                'SGP (500)': [
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_27-09-2022_23_18',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_06-10-2022_17_37',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_25-10-2022_10_19',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_30-10-2022_19_01',
                    'experiment_3/so/SGPPushSumAgent_500A_100E_50B_sparse(directed-3)_01-11-2022_19_41',
                ],
                'D$^2$ (100)': [
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_17-09-2022_19_33',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_01-10-2022_19_06',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_03-10-2022_02_15',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_15-10-2022_10_13',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_18-10-2022_20_38',
                ],
                'SGP (100)': [
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_17_26',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_20_12',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_30-09-2022_01_15',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_20_52',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_13-10-2022_17_08',
                ],
            }
        },
    }
    fig, axs = side_by_side(viz, fig_size=(7, 3))
    axs[0].legend().remove()
    axs[1].legend(bbox_to_anchor=(1.01, 0.98))
    ratio = 0.72
    box0 = axs[0].get_position()
    axs[0].set_position([box0.x0, box0.y0, box0.width * ratio, box0.height])
    box = axs[1].get_position()
    axs[1].set_position([box.x0 - box0.width * (1 - ratio), box.y0, box.width * ratio, box.height])


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
    """
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for a_acc in [parse_timeline(None, val, agg_fn=lambda x: np.array(x))[1] for val in v]:
            a_acc = np.array(a_acc)[20:]
            a_max = np.max(a_acc, axis=0)
            accs[k].append(a_max)
        accs[k] = np.array(accs[k]).flatten()

    half_len = int(len(accs)/2)
    names = list(accs.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i+half_len], ttest_ind(accs[names[i]], accs[names[i+half_len]]))
    """
    # 5 points
    accs = {}
    for k, v in viz.items():
        accs[k] = []
        for avg_acc in [parse_timeline(None, val)[1] for val in v]:
            avg_acc = np.array(avg_acc)[20:]
            avg_max = np.max(avg_acc)
            accs[k].append(avg_max)
        accs[k] = np.array(accs[k]).flatten()

    half_len = int(len(accs) / 2)
    names = list(accs.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i + half_len], ttest_ind(accs[names[i]], accs[names[i + half_len]]))

    """
    accs = []
    for k, v in viz.items():
        _, acc, _ = parse_timeline(None, v)
        accs.append(acc)

    half_len = int(len(accs)/2)
    names = list(viz.keys())
    for i in range(half_len):
        print(names[i], "<->", names[i+half_len], mannwhitneyu(accs[i][20:], accs[i+half_len][20:]))

    print('-------------------')
    print('Directed')
    dir_ind = [0, 1, 4, 5]
    for i in dir_ind:
        for j in dir_ind:
            print(names[i], "<->", names[j], mannwhitneyu(accs[i][20:], accs[j][20:]))

    print('-------------------')
    print('Undirected')
    undir_ind = [2, 3, 6, 7]
    for i in undir_ind:
        for j in undir_ind:
            print(names[i], "<->", names[j], mannwhitneyu(accs[i][20:], accs[j][20:]))

    print("Accuracy gain")
    for i in range(len(names)):
        print(names[i], max(accs[i]))
    """


def independent_acc():
    ds_viz = {
        'Reddit': {
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
        },
        'StackOverflow': {
            'agents': [
                'experiment_2/indp/so/P2PAgent_100A_100E_50B_4V_10-10-2022_02_09',
                'experiment_2/indp/so/P2PAgent_100A_100E_50B_4V_10-10-2022_03_06',
                'experiment_2/indp/so/P2PAgent_100A_100E_50B_4V_10-10-2022_02_31',
            ],
            'single': [
                'experiment_2/indp/so/Single_100A_100E_50B_4V_10-10-2022_07_35',
                'experiment_2/indp/so/Single_100A_100E_50B_4V_10-10-2022_07_36',
                'experiment_2/indp/so/Single_100A_100E_50B_4V_10-10-2022_07_58',
            ]
        }
    }

    for ds, viz in ds_viz.items():
        print(ds)
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


def review_exp_ring():
    viz = {
        'MNIST-IID (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_08-07-2022_03_44',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-07-2022_12_35',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-07-2022_12_20',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_08-07-2022_06_08',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_13-07-2022_10_55',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_13-07-2022_11_09'
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_08-07-2022_03_47',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_03_09',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-07-2022_03_17',
                ],
            },
        },
        'MNIST-Pathological (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-09-2022_15_36',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-09-2022_15_22',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_13-09-2022_15_36',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-09-2022_00_21',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-09-2022_17_06',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_14-09-2022_17_17',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-09-2022_18_01',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-09-2022_17_59',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_14-09-2022_18_02',
                ],
            },
        },
        'MNIST-Practical (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_16-09-2022_20_56',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_16-09-2022_21_37',
                    'rev/D2Agent_100A_100E_32B_ring(undirected-2)_16-09-2022_21_37',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_17-09-2022_21_51',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_17-09-2022_21_54',
                    'rev/GoSGDAgent_100A_100E_32B_ring(undirected-2)_17-09-2022_21_59',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_17-09-2022_14_20',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_17-09-2022_14_29',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(undirected-2)_17-09-2022_14_29',
                ],
            },
        },
        'Reddit (Unirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_02',
                    'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_22',
                    'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_18',
                    # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_16_06',
                    # 'experiment_1/undirected/nonbn/D2Agent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_12_16',
                ],
                'GoSGD': [
                    'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_35',
                    'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_09_16',
                    'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_20-12-2021_11_20',
                    # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_14_17',
                    # 'experiment_1/undirected/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_34',
                ],
                'SGP': [
                    'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_29-11-2021_12_29',
                    'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_18_32',
                    'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_27-11-2021_22_55',
                    # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_15_49',
                    # 'experiment_1/undirected/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(undirected)_N100_NB2_TV-1_11-03-2022_11_57',
                ],
            },
        },
        'MNIST-IID (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_08-07-2022_04_03',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-07-2022_12_23',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-07-2022_12_22',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_08-07-2022_05_43',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-07-2022_10_45',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-07-2022_11_12',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_08-07-2022_05_10',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_14-07-2022_03_04',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_14-07-2022_03_03',
                ],
            },
        },
        'MNIST-Pathological (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-09-2022_15_59',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-09-2022_15_56',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_13-09-2022_15_50',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-09-2022_14_02',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-09-2022_13_43',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_13-09-2022_13_38',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_13-09-2022_15_41',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_13-09-2022_15_23',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_13-09-2022_15_27',
                ],
            },
        },
        'MNIST-Practical (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_18-09-2022_06_05',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_18-09-2022_06_05',
                    'rev/D2Agent_100A_100E_32B_ring(directed-1)_18-09-2022_06_21',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_18-09-2022_19_41',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_18-09-2022_19_36',
                    'rev/GoSGDAgent_100A_100E_32B_ring(directed-1)_19-09-2022_07_18',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_19-09-2022_02_40',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_19-09-2022_02_47',
                    'rev/SGPPushSumAgent_100A_100E_32B_ring(directed-1)_19-09-2022_02_54',
                ],
            },
        },
        'Reddit (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_36',
                    'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_08_07',
                    'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_18-12-2021_09_30',
                    # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_08_04',
                    # 'experiment_1/directed/nonbn/D2Agent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_11',
                ],
                'GoSGD': [
                    'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_58',
                    'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_20-12-2021_08_38',
                    'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_21-12-2021_04_26',
                    # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_06_02',
                    # 'experiment_1/directed/nonbn/GoSGDAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_03_44',
                ],
                'SGP': [
                    'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_46',
                    'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_17_25',
                    'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_17-12-2021_19_16',
                    # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_07_43',
                    # 'experiment_1/directed/nonbn/SGPPushSumAgent_100A_100E_50B_2V_ring(directed)_N100_NB1_TV-1_12-03-2022_04_26',
                ],
            },
        }
    }
    side_by_side(viz, n_rows=2, fig_size=(15, 7),
                 axis_lim=[{'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]},
                           {'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]}])

    print("Statistics")
    data = {}
    for _, g in viz.items():
        for k, v in g['viz'].items():
            if k in data:
                data[k + '-' + _].extend(v)
            else:
                data[k + '-' + _] = v
    res = {}
    for k, v in data.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms',
                                         metric='test_model-accuracy_no_oov' if 'Reddit' in k else 'test_model-sparse_categorical_accuracy')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))
        k1 = k.split(' ')[0]
        if k1 in res:
            res[k1].append(round(acc / len(v), 2))
        else:
            res[k1] = [round(acc / len(v), 2)]
    print('---------------------')
    for k, v in res.items():
        print(k, round(np.mean(v), 2))


def review_exp_sparse():
    viz = {
        'MNIST-IID (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_09-07-2022_03_38',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_18',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_11',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_08-07-2022_10_21',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_07_35',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_07_46',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_08-07-2022_10_40',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_49',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_16-07-2022_09_22',
                ],
            },
        },
        'MNIST-Pathological (Unirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_14-09-2022_16_03',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_14-09-2022_16_06',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_14-09-2022_16_04',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_14-09-2022_14_55',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_14-09-2022_14_52',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_14-09-2022_14_30',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_14-09-2022_15_54',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_14-09-2022_16_00',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_14-09-2022_15_54',
                ],
            },
        },
        'MNIST-Practical (Unirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_19-09-2022_23_27',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_19-09-2022_23_30',
                    'rev/D2Agent_100A_100E_32B_sparse(undirected-3)_19-09-2022_23_36',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_18-09-2022_21_16',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_19-09-2022_01_27',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(undirected-3)_19-09-2022_01_30',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_19-09-2022_22_35',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_20-09-2022_11_38',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(undirected-3)_19-09-2022_22_27',
                ],
            },
        },
        'Reddit (Undirected)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_08_01',
                    'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_05_42',
                    'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_12_07',
                    # 'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_09_18',
                    # 'rev/D2Agent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_07',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_17-07-2022_23_07',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_17-07-2022_22_19',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_01_01',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_09',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_12_53',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_02_10',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_26',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_02_15',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_18-07-2022_00_45',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(undirected-3)_17-07-2022_20_02',
                ],
            }
        },
        'MNIST-IID (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_09-07-2022_03_18',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_59',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_45',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_14',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_51',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_41',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_34',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_38',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_03_41',
                ],
            },
        },
        'MNIST-Pathological (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_53',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_49',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_57',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_35',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_28',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_48',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_54',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_56',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_48',
                ],
            },
        },
        'MNIST-Practical (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_20-09-2022_13_11',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_21-09-2022_20_27',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_21-09-2022_20_27',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_41',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_39',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_42',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_20-09-2022_11_09',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_21-09-2022_00_29',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_21-09-2022_19_51',
                ],
            },
        },
        'Reddit (Directed)': {
            'x_axis': 'epoch',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_50B_sparse(directed-3)_18-07-2022_17_58',
                    'rev/D2Agent_100A_100E_50B_sparse(directed-3)_18-07-2022_17_13',
                    'rev/D2Agent_100A_100E_50B_sparse(directed-3)_18-07-2022_20_59',
                    # 'rev/D2Agent_100A_100E_50B_sparse(directed-3)_19-07-2022_16_03',
                    # 'rev/D2Agent_100A_100E_50B_sparse(directed-3)_19-07-2022_01_54',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_08_24',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_07_29',
                    'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_11_22',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_09_08',
                    # 'rev/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_06_27',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_06_47',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_06_21',
                    'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_10_05',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_09_10',
                    # 'rev/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_17-07-2022_05_24',
                ],
            }
        },
    }
    side_by_side(viz, n_rows=2, fig_size=(15, 7),
                 axis_lim=[{'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]},
                           {'y': [85, 100]}, {'y': [85, 100]}, {'y': [85, 100]}, {'y': [0, 10]}])

    print("Statistics")
    data = {}
    for _, g in viz.items():
        for k, v in g['viz'].items():
            if k in data:
                data[k + '-' + _].extend(v)
            else:
                data[k + '-' + _] = v
    res = {}
    for k, v in data.items():
        t = 0
        acc = 0
        for vl in v:
            t1, t_acc = resolve_timeline(vl, 'comms',
                                         metric='test_model-accuracy_no_oov' if 'Reddit' in k else 'test_model-sparse_categorical_accuracy')
            t1 = t1[:101]
            t += t1[-1] / 100
            acc += max(t_acc)
        print(k, "# msgs:", round(t / len(v)), "Avg top acc:", round(acc / len(v), 2))
        k1 = k.split(' ')[0]
        if k1 in res:
            res[k1].append(round(acc / len(v), 2))
        else:
            res[k1] = [round(acc / len(v), 2)]
    print('---------------------')
    for k, v in res.items():
        print(k, round(np.mean(v), 2))


def prez_info_mnist():
    viz = {
        'MNIST-IID': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_09-07-2022_03_18',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_59',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_45',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_14',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_51',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_41',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_34',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_38',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_03_41',
                ],
            },
        },
        'MNIST-Pathological': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_53',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_49',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_57',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_35',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_28',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_48',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_54',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_56',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_48',
                ],
            },
        },
        'MNIST-Practical': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_20-09-2022_13_11',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_21-09-2022_20_27',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_21-09-2022_20_27',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_41',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_39',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_42',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_20-09-2022_11_09',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_21-09-2022_00_29',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_21-09-2022_19_51',
                ],
            },
        },

        'MNIST-IID (uniform)': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_09-07-2022_03_18',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_59',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_45',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_14',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_51',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_02_41',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_08-07-2022_10_34',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_04_38',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_17-07-2022_03_41',
                ],
            },
        },
        'MNIST-Pathological (uniform)': {
            'x_axis': 'epoch',
            'metric': 'val_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_53',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_49',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_57',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_35',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_28',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_17_48',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_54',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_56',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_14-09-2022_18_48',
                ],
            },
        },
        'MNIST-Practical (uniform)': {
            'x_axis': 'epoch',
            'metric': 'val_model-sparse_categorical_accuracy',
            'viz': {
                'D$^2$': [
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_20-09-2022_13_11',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_21-09-2022_20_27',
                    'rev/D2Agent_100A_100E_32B_sparse(directed-3)_21-09-2022_20_27',
                ],
                'GoSGD': [
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_41',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_39',
                    'rev/GoSGDAgent_100A_100E_32B_sparse(directed-3)_19-09-2022_21_42',
                ],
                'SGP': [
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_20-09-2022_11_09',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_21-09-2022_00_29',
                    'rev/SGPPushSumAgent_100A_100E_32B_sparse(directed-3)_21-09-2022_19_51',
                ],
            },
        },
    }
    side_by_side(viz, n_rows=2, fig_size=(15, 7),
                 axis_lim=[{'y': [85, 100]}, {'y': [40, 100], 'step': 5}, {'y': [85, 100]},
                           {'y': [85, 100]}, {'y': [40, 100], 'step': 5}, {'y': [85, 100]}])


def pre_info_nwp():
    viz = {
        'Reddit': {
            'x_axis': 'epoch',
            # 'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_01_19',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_06_33',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_13-03-2022_00_54',
                    'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-03-2022_23_26',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_13_27',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_08_51',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_38',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_03_49',
                    'experiment_2/fixed/directed/GoSGDAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_09',
                ],
                'D$^2$': [
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_11_56',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_06',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_11_49',
                    'experiment_2/fixed/directed/D2Agent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_06_13',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_40',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_12_01',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_30-11-2021_14_51',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_09_19',
                    'experiment_2/fixed/directed/SGPPushSumAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-03-2022_04_48',
                ],
            }
        },
        'StackOverflow': {
            'x_axis': 'epoch',
            # 'colors': colors,
            'viz': {
                ALG_NAME: [
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_22_25',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_27-09-2022_20_15',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_09_10',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_07-10-2022_22_56',
                    'experiment_2/fixed/directed/so/P2PAgent_100A_100E_50B_sparse(directed-3)_08-10-2022_14_22',
                ],
                'GoSGD': [
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_17-09-2022_14_16',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_01-10-2022_13_02',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_02-10-2022_19_35',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_16_31',
                    'experiment_2/fixed/directed/so/GoSGDAgent_100A_100E_50B_sparse(directed-3)_12-10-2022_21_49',
                ],
                'D$^2$': [
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_17-09-2022_19_33',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_01-10-2022_19_06',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_03-10-2022_02_15',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_15-10-2022_10_13',
                    'experiment_2/fixed/directed/so/D2Agent_100A_100E_50B_sparse(directed-3)_18-10-2022_20_38',
                ],
                'SGP': [
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_18-09-2022_17_26',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_28-09-2022_20_12',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_30-09-2022_01_15',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_20_52',
                    'experiment_2/fixed/directed/so/SGPPushSumAgent_100A_100E_50B_sparse(directed-3)_13-10-2022_17_08',
                ],
            }
        },
    }

    side_by_side(viz, fig_size=(7, 7 / 2), axis_lim=[{'y': [0, 11]}, {'y': [0, 14]}])


def prez_info_abl():
    side_by_side({
        'Reddit': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (BN-FT)': [
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_53',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_10_56',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_04_15',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_23',
                    'exp_es/es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_40',
                ],
                ALG_NAME + ' (No BN-FT)': [
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_02_41',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_01',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_19-01-2022_03_24',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_12_22',
                    'exp_es/no_es/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_09-03-2022_11_31',
                ],
            }
        },
        'StackOverflow': {
            'x_axis': 'epoch',
            'colors': ['r', 'orange', 'b', 'g'],
            'viz': {
                ALG_NAME + ' (BN-FT)': [
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_15-09-2022_04_28',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_10_45',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_17_26',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_07-10-2022_23_19',
                    'exp_es/es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_08-10-2022_20_35',
                ],
                ALG_NAME + ' (No BN-FT)': [
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_15-09-2022_03_55',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_00_42',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_19-09-2022_13_32',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_10-10-2022_20_12',
                    'exp_es/no_es/so/P2PAgent_100A_100E_50B_sparse(directed-3)_11-10-2022_19_14',
                ],
            }
        },
    }, fig_size=(7, 7/2), axis_lim=[{'y': [0, 11]}, {'y': [0, 14]}])


if __name__ == '__main__':
    prez_info_abl()
