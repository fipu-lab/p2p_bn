import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import json

LABELS = {
    "epoch": "Epoch",
    "comms": "# of Messages",
    "acomms": "# of Agent Messages",
    "examples": "Examples",
    "round": "Round"
}


def read_json(filename):
    with open('log/' + filename + '.json', "r") as infile:
        f_json = json.loads(infile.read())
    return f_json


def read_graph(filename):
    return read_json(filename)['graph']


def read_agents(filename):
    return read_json(filename)['agents']


def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # '%.2f%s'
    return '%i%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def calc_agent_timeline(data, x_axis, agg_fn):
    acc, v_acc, t_acc = [], [], []
    mk = 'model-accuracy_no_oov'
    rounds = list(range(len(data[list(data.keys())[0]]['train_' + mk])))
    total_examples = sum([data[a_key]['train_len'] for a_key in list(data.keys())])
    x_time = [] if x_axis != 'Round' else rounds
    for rind in rounds:
        test = [data[a_id]['test_' + mk][rind] for a_id in data.keys()]
        t_acc.append(agg_fn(test) * 100)

        examples = sum([data[a_key]['examples'][rind] for a_key in list(data.keys())])
        if x_axis == 'epoch':
            x_time.append(round(examples / total_examples))
        elif x_axis == 'examples':
            x_time.append(examples)
        elif x_axis == 'comms':
            comms = sum([sum(data[a_key]['useful_msg'][:rind] + data[a_key]['useless_msg'][:rind]) for a_key in
                         list(data.keys())])
            x_time.append(comms)
        elif x_axis == 'acomms':
            acomms = np.average([sum(data[a_key]['useful_msg'][:rind] + data[a_key]['useless_msg'][:rind]) for a_key in
                                 list(data.keys())])
            x_time.append(acomms)
    return x_time, t_acc


def calc_fl_timeline(data, x_axis, agg_fn):
    t_acc = []
    x_time = [] if x_axis != 'Round' else [int(k) for k in data.keys()]
    for dk, dv in data.items():
        # v_acc.append(agg_fn(dv['Valid']) * 100)
        t_acc.append(agg_fn(dv['Test']) * 100)

        # acc.append(agg_fn(np.average([dv['Valid'], dv['Test']], axis=0)) * 100)
        if x_axis == 'epoch':
            x_time.append(dv['Epoch'])
        elif x_axis == 'examples':
            x_time.append(dv['Examples'])
        elif x_axis == 'comms':
            # Double the communications because server needs to send and receive message
            x_time.append(dv['Sampled_Clients'] * 2)
    return x_time, t_acc


def resolve_timeline(filename, x_axis, agg_fn=np.average):
    data = read_json(filename)
    if 'agents' in data:
        return calc_agent_timeline(data['agents'], x_axis, agg_fn)
    else:
        return calc_fl_timeline(data, x_axis, agg_fn)


def parse_timeline(name, filename, x_axis='Examples', agg_fn=np.average):
    if isinstance(filename, str):
        t, a = resolve_timeline(filename, x_axis, agg_fn)
        return t, a, None
    if isinstance(filename, list):
        time_t, acc_t = None, None
        accs = []
        for fl in filename:
            time, acc = resolve_timeline(fl, x_axis, agg_fn)
            accs.append(acc)
            if acc_t is None:
                acc_t = acc
                time_t = time
            else:
                acc_t = np.add(acc_t, acc)
                time_t = np.add(time_t, time)

        return np.array(time_t) / float(len(filename)), np.array(acc_t) / float(len(filename)), accs


def calc_fill_between(accs):
    accs = np.array(accs)
    min_vals = [min(accs[:, i]) for i in range(accs.shape[1])]
    max_vals = [max(accs[:, i]) for i in range(accs.shape[1])]
    return min_vals, max_vals


def plot_items(ax, x_axis, viz_dict, title=None, agg_fn=np.average):
    legend = []
    x_axis2 = None
    if isinstance(x_axis, list):
        x_axis, x_axis2 = x_axis

    for k, v in viz_dict.items():
        x_time, t_acc, accs = parse_timeline(k, v, x_axis, agg_fn)
        ax.plot(x_time, t_acc)
        if accs is not None:
            min_acc, max_acc = calc_fill_between(accs)
            ax.fill_between(x_time, max_acc, min_acc,  alpha=0.2)
        legend.append(k)
        if x_axis2 is not None:
            x_time2, t_acc2, accs2 = parse_timeline(k, v, x_axis2, agg_fn)
            xmin, xmax = x_time[0], x_time[-1]

            def x_lim_fn(x):
                left_pct = (xmin - x[0]) / xmax
                right_pct = (x[1] - xmax) / xmax
                left = x_time2[0] - left_pct * x_time2[-1]
                right = right_pct * x_time2[-1] + x_time2[-1]
                return [left, right]

            ax2 = ax.secondary_xaxis('top', functions=(x_lim_fn, lambda x: x))
            ax2.xaxis.set_major_formatter(FuncFormatter(human_format))
            ax2.set_xlabel(LABELS[x_axis2])

    ax.set_xlabel(LABELS[x_axis])
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.set_ylabel('Test UA (%)')
    ax.grid()
    ax.legend(legend, loc='lower right')
    ax.yaxis.set_major_locator(MultipleLocator())
    if title:
        ax.set_title(title)


def show(viz_dict, x_axises=tuple(['comms']), agg_fn=np.average):
    fig, axs = plt.subplots(1, len(x_axises))
    if len(x_axises) < 2:
        axs = [axs]
    for ax, x_axis in zip(axs, x_axises):
        plot_items(ax, x_axis, viz_dict, None, agg_fn)
    # plt.savefig('/Users/robert/Desktop/acomms.svg', format='svg', dpi=1200)
    plt.show()


def side_by_side(viz_dict, agg_fn=np.average, fig_size=(10, 5), n_rows=1):
    fig, axs = plt.subplots(n_rows, int(len(viz_dict) / n_rows))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()
    for ax, (plot_k, plot_v) in zip(axs, viz_dict.items()):
        plot_items(ax, plot_v['x_axis'], plot_v['viz'], plot_k, agg_fn)
    max_y = round(max([ax.get_ylim()[1] for ax in axs]))
    for ax in axs:
        ax.set_ylim([0, max_y])
    # plt.savefig('/Users/robert/Desktop/test.svg', format='svg', dpi=300)
    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])
    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
    plt.show()


def plot_graph(viz_dict, fig_size=(10, 5), n_rows=1, node_size=300):
    import networkx as nx
    from p2p.graph_manager import nx_graph_from_saved_lists
    fig, axs = plt.subplots(n_rows, int(len(viz_dict) / n_rows))
    axs = axs.flatten()
    if len(viz_dict) < 2:
        axs = [axs]
    for ax, (k, v) in zip(axs, viz_dict.items()):
        nx_graph = nx_graph_from_saved_lists(read_graph(v), directed='(directed)' in v)
        for i in range(nx_graph.number_of_nodes()):
            if nx_graph.has_edge(i, i):
                nx_graph.remove_edge(i, i)
        ax.set_title(k, fontsize=18)
        # pos = nx.spring_layout(nx_graph)
        pos = None
        nx.draw(nx_graph, pos=pos, ax=ax, node_color='b', node_size=node_size)

    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])
    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
    plt.show()


if __name__ == '__main__':
    show({
        'decay(0.0045) - 100': 'experiment_3/decaying_45_P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_22-12-2021_12_41',
        'decay(0.009) - 100': 'experiment_3/decaying_9_P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_22-12-2021_16_06',
        'decay - 100': 'experiment_3/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_14-12-2021_00_22',

        # 'fixed - 500': 'experiment_3/old_P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_04-12-2021_16_20',
        # 'decaying': 'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_14-12-2021_15_37',
        # 'fixed-1000': 'experiment_3/old_P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_09-12-2021_04_28',
        # 'decay-1000': 'experiment_3/P2PAgent_1000A_100E_50B_4V_sparse(directed)_N1000_NB3_TV-1_19-12-2021_16_30',
        # 'Fixed-100': 'experiment_2/fixed/directed/P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_29-11-2021_20_59',
        # 'Decaying-100': 'P2PAgent_100A_100E_50B_4V_sparse(directed)_N100_NB3_TV-1_10-12-2021_21_37',
        # 'Fix(0.99)-100': 'P2PAgent_100A_99E_50B_4V_sparse(directed)_N100_NB3_TV-1_12-12-2021_22_24',
        # 'Fixed-500': 'experiment_3/P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_04-12-2021_16_20',
        # 'FL-500': 'experiment_3/fl_500C_10TR_2V(0_005S-0_005C)_100E_num_examples_06-12-2021_12_24',
        # 'Decaying-500': 'P2PAgent_500A_100E_50B_4V_sparse(directed)_N500_NB3_TV-1_11-12-2021_16_03',

    },
        x_axises=['epoch'],
        # 'round', 'examples', 'epoch', 'comms', 'acomms'
    )
