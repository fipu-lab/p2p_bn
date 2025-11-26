from plot.visualize import side_by_side, plt, resolve_timeline, parse_timeline
from scipy.stats import ttest_ind
import numpy as np


def add_titles(fig):
    fig.subplots_adjust(hspace=0.6, top=0.9)
    font = {"color": "black", "weight": "heavy", "size": 20}
    box_style = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # titles = ("Multi-task without BERT freezing", "Multi-task with BERT freezing")
    pad = " " * 100  # 83
    titles = (pad + "  Sparse  " + pad, pad + "MT-EF" + pad, pad + "MT-AVG" + pad)
    fig.text(x=0.5, y=0.94, s=titles[0], fontdict=font, bbox=box_style, ha="center")
    fig.text(x=0.5, y=0.61, s=titles[1], fontdict=font, bbox=box_style, ha="center")
    fig.text(x=0.5, y=0.285, s=titles[2], fontdict=font, bbox=box_style, ha="center")


colors = ["r", "g", "b", "indigo"]  # , 'orange']
viz = {
    "Reddit (MTP)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "reddit-bert-nwp->test_model-sparse_categorical_accuracy",
        "viz": {
            "Reddit": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01",
            ],
            "Reddit+StackOverflow": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_44",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_06_39",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_19_11",
            ],
            "Reddit+CoNNL": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_18_54",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_03_21",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_01_29",
            ],
            "Reddit+Few-NERD": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_07_57",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_08_00",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_06_07",
            ],
        },
    },
    "StackOverflow (MTP)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy",
        "viz": {
            "Stackoverflow": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11",
            ],
            "StackOverflow+Reddit": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_44",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_06_39",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_19_11",
            ],
            "StackOverflow+CoNNL": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_14_33",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_11_40",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_10_57",
            ],
            "StackOverflow+Few-NERD": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_21_07",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_05_52",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_01_55",
            ],
        },
    },
    "CoNNL (NER)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "conll-bert-ner->test_model-macro_avg_f1_score",
        "viz": {
            "CoNLL": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27",
            ],
            "CoNNL+Reddit": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_18_54",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_03_21",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_20-01-2023_01_29",
            ],
            "CoNNL+StackOverflow": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_14_33",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_11_40",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_10_57",
            ],
            "CoNNL+Few-NERD": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_07_39",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_07_20",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_09_06",
            ],
        },
    },
    "Few-NERD (NER)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "few_nerd-bert-ner->test_model-macro_avg_f1_score",
        "viz": {
            "Few-NERD": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28",
            ],
            "Few-NERD+Reddit": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_06-01-2023_07_57",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_08_00",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_19-01-2023_06_07",
            ],
            "Few-NERD+StackOverflow": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_21_07",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_05_52",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_01_55",
            ],
            "Few-NERD+CoNNL": [
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_07_39",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_07_20",
                "mt/cluster/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_18-01-2023_09_06",
            ],
        },
    },
    #  ---- MT ----
    "Reddit (MTP) ": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "reddit-bert-nwp->test_model-sparse_categorical_accuracy",
        "viz": {
            "Reddit": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01",
            ],
            "Reddit+StackOverflow": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_39",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_51",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_07_00",
            ],
            "Reddit+CoNNL": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_18_28",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_17-01-2023_01_59",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_39",
            ],
            "Reddit+Few-NERD": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_22_59",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_15_21",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_18_02",
            ],
        },
    },
    "StackOverflow (MTP) ": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy",
        "viz": {
            "Stackoverflow": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11",
            ],
            "StackOverflow+Reddit": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_23_39",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_51",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_07_00",
            ],
            "StackOverflow+CoNNL": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_11_16",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_23_25",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_22_25",
            ],
            "StackOverflow+Few-NERD": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_03_31",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_20_10",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_19_23",
            ],
        },
    },
    "CoNNL (NER) ": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "conll-bert-ner->test_model-macro_avg_f1_score",
        "viz": {
            "CoNLL": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27",
            ],
            "CoNNL+Reddit": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_18_28",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_17-01-2023_01_59",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_22_39",
            ],
            "CoNNL+StackOverflow": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_11_16",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_23_25",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_22_25",
            ],
            "CoNNL+Few-NERD": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_13_18",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_03",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_16",
            ],
        },
    },
    "Few-NERD (NER) ": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "few_nerd-bert-ner->test_model-macro_avg_f1_score",
        "viz": {
            "Few-NERD": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28",
            ],
            "Few-NERD+Reddit": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_04-01-2023_22_59",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_15_21",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_16-01-2023_18_02",
            ],
            "Few-NERD+StackOverflow": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_07-01-2023_03_31",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_20_10",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_19_23",
            ],
            "Few-NERD+CoNNL": [
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_05-01-2023_13_18",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_03",
                "mt/cluster/mt/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_15-01-2023_09_16",
            ],
        },
    },
    #  ---- AVG ----
    " Reddit (MTP)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "reddit-bert-nwp->test_model-sparse_categorical_accuracy",
        "viz": {
            "Reddit": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_26",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_20_31",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_18_01",
            ],
            "Reddit+StackOverflow": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_26-11-2025_00_14",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_20_28",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_17_55",
            ],
            "Reddit+CoNNL": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_19_08",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_20_02",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_14_57",
            ],
            "Reddit+Few-NERD": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_11_24",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_12_08",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_12_07",
            ],
        },
    },
    " StackOverflow (MTP)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "stackoverflow-bert-nwp->test_model-sparse_categorical_accuracy",
        "viz": {
            "Stackoverflow": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_30-12-2022_22_05",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_13_13",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_12_11",
            ],
            "StackOverflow+Reddit": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_26-11-2025_00_14",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_20_28",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_17_55",
            ],
            "StackOverflow+CoNNL": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_04_50",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_02_36",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_22-11-2025_23_35",
            ],
            "StackOverflow+Few-NERD": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_13_04",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_11_26",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_10_59",
            ],
        },
    },
    " CoNNL (NER)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "conll-bert-ner->test_model-macro_avg_f1_score",
        "viz": {
            "CoNLL": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_31-12-2022_02_41",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_18_07",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_08-01-2023_02_27",
            ],
            "CoNNL+Reddit": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_19_08",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_20_02",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_24-11-2025_14_57",
            ],
            "CoNNL+StackOverflow": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_04_50",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_02_36",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_22-11-2025_23_35",
            ],
            "CoNNL+Few-NERD": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_04_55",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_04_59",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_05_11",
            ],
        },
    },
    " Few-NERD (NER)": {
        "x_axis": "epoch",
        "colors": colors,
        "metric": "few_nerd-bert-ner->test_model-macro_avg_f1_score",
        "viz": {
            "Few-NERD": [
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_01-01-2023_02_25",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_32",
                "mt/ind/BertAgent_20A_300E_50B_sparse(directed-3)_07-01-2023_16_28",
            ],
            "Few-NERD+Reddit": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_11_24",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_12_08",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_12_07",
            ],
            "Few-NERD+StackOverflow": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_13_04",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_11_26",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_21-11-2025_10_59",
            ],
            "Few-NERD+CoNNL": [
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_04_55",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_04_59",
                "mt/avg/BertAgent_40A_300E_50B_sparse_clusters(directed-2)_23-11-2025_05_11",
            ],
        },
    },
}


def skip_first(_vv):
    del _vv["viz"][list(_vv["viz"].keys())[0]]
    return _vv


# viz = {vk: skip_first(vv) for vk, vv in viz.items()}

fig, axs = side_by_side(
    viz,
    axis_lim=[
        {"y": [4, 12], "step": 1},
        {"y": [4, 13], "step": 1},
        {"y": [40, 57], "step": 2},
        {"y": [20, 34], "step": 2},
        {"y": [4, 12], "step": 1},
        {"y": [4, 13], "step": 1},
        {"y": [40, 57], "step": 2},
        {"y": [20, 34], "step": 2},
        {"y": [4, 12], "step": 1},
        {"y": [4, 13], "step": 1},
        {"y": [40, 57], "step": 2},
        {"y": [20, 34], "step": 2},
    ],
    n_rows=3,
    fig_size=(9 * 2, 9 * 1.5),
)
add_titles(fig)

# """
print("Statistic")
for vk, vv in viz.items():
    print(vk)
    baseline = None
    baseline_max_a = None
    for k, v in vv["viz"].items():
        t, accs = parse_timeline(None, v, x_axis="examples", metric=vv["metric"])[1:]
        max_a = round(max(t), 2)
        if baseline is None:
            baseline_max_a = max_a
            baseline = t[40:]
            print("\t", k, "{:.2f}".format(max_a) + "\\%")
        else:
            rel_inc = round((max_a - baseline_max_a) / baseline_max_a * 100, 2)
            p_val = ttest_ind(baseline, t[40:])[1]

            p_text = "{} {}".format(
                "{:.2f}".format(max_a) + "\\%",
                "({}\\%)".format(
                    "+" + "{:.2f}".format(rel_inc)
                    if rel_inc > 0
                    else "{:.2f}".format(rel_inc)
                ),
            )
            if rel_inc > 0:
                p_text = "\\textbf{" + p_text + "}"
            if p_val < 0.05:
                p_text += " \\textbf{**}"
            print("\t", k, p_text)

fig.savefig("plot/fig.pdf")
