import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import string
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


def calc_agent_timeline(data, x_axis, agg_fn, metric='test_model-accuracy_no_oov'):
    acc, v_acc, t_acc = [], [], []

    rounds = list(range(len(data[list(data.keys())[0]]["examples"])))
    total_examples = sum([data[a_key]['train_len'] for a_key in list(data.keys())])
    x_time = [] if x_axis != 'Round' else rounds

    dataset_name = None
    if '->' in metric:
        dataset_name = metric.split('->')[0]
        metric = metric.split('->')[-1]

    for rind in rounds:
        test = [data[a_id][metric][rind] for a_id in data.keys() if metric in data[a_id] and (dataset_name is None or dataset_name == data[a_id]['dataset_name'])]
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


def resolve_timeline(filename, x_axis, agg_fn=np.average, metric='test_model-accuracy_no_oov'):
    data = read_json(filename)
    if 'agents' in data:
        return calc_agent_timeline(data['agents'], x_axis, agg_fn, metric)
    else:
        return calc_fl_timeline(data, x_axis, agg_fn)


def parse_timeline(name, filename, x_axis='Examples', agg_fn=np.average, metric='test_model-accuracy_no_oov'):
    if isinstance(filename, str):
        t, a = resolve_timeline(filename, x_axis, agg_fn, metric)
        return t, a, None
    if isinstance(filename, list):
        time_t, acc_t = None, None
        accs = []
        for fl in filename:
            time, acc = resolve_timeline(fl, x_axis, agg_fn, metric)
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


def plot_items(ax, x_axis, viz_dict, title=None, y_label='Test UA (%)', colors=None, agg_fn=np.average, metric='test_model-accuracy_no_oov'):
    legend = []
    x_axis2 = None
    if isinstance(x_axis, list):
        x_axis, x_axis2 = x_axis

    for i, (k, v) in enumerate(viz_dict.items()):
        x_time, t_acc, accs = parse_timeline(k, v, x_axis, agg_fn, metric[i] if isinstance(metric, list) else metric)
        # ax.plot(x_time, t_acc)
        args = {}
        if colors is not None:
            args['color'] = colors[i]
        ax.plot(x_time, t_acc, **args, label=k, linewidth=1)
        if accs is not None:
            min_acc, max_acc = calc_fill_between(accs)
            ax.fill_between(x_time, max_acc, min_acc, alpha=0.05, **args)
        legend.append(k)
        if x_axis2 is not None:
            x_time2, t_acc2, accs2 = parse_timeline(k, v, x_axis2, agg_fn,  metric[i] if isinstance(metric, list) else metric)
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

    # ax.set_xlabel(LABELS[x_axis] + "\n" + string.ascii_lowercase[ax.get_subplotspec().num1] + ")")
    gs = ax.get_subplotspec().get_gridspec()
    if gs.ncols * gs.nrows > 1:
        ax.set_xlabel(LABELS[x_axis] + "\n" + string.ascii_lowercase[ax.get_subplotspec().num1] + ")")
    else:
        ax.set_xlabel(LABELS[x_axis])
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.set_ylabel(y_label)
    ax.grid()
    ax.legend(legend, loc='lower right')
    # ax.legend(legend)
    """
    ax.set_xlabel(LABELS[x_axis])
    if title == 'IID' or title == 'praktični ne-IID':
        ax.legend(legend, loc='lower right')
    elif title == 'patološki ne-IID':
        ax.legend(legend, loc='center right')
    # """
    ax.yaxis.set_major_locator(MultipleLocator())
    # ax.text(0.5, -.5, string.ascii_lowercase[ax.get_subplotspec().colspan.start] + ")")
    if title:
        ax.set_title(title)


def show(viz_dict, x_axises=tuple(['comms']), agg_fn=np.average, metric='test_model-accuracy_no_oov'):
    fig, axs = plt.subplots(1, len(x_axises))
    if len(x_axises) < 2:
        axs = [axs]
    for ax, x_axis in zip(axs, x_axises):
        plot_items(ax, x_axis, viz_dict, None, agg_fn=agg_fn, metric=metric)
    max_y = round(max([ax.get_ylim()[1] for ax in axs]))
    for ax in axs:
        ax.set_ylim([0, max_y])
    # plt.savefig('/Users/robert/Desktop/acomms.svg', format='svg', dpi=1200)
    plt.show()


def side_by_side(viz_dict, agg_fn=np.average, fig_size=(10, 5), n_rows=1, axis_lim=None):
    # plt.rcParams.update({'font.size': 11})
    fig, axs = plt.subplots(n_rows, int(len(viz_dict) / n_rows))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()
    for ax, (plot_k, plot_v) in zip(axs, viz_dict.items()):
        metrics = plot_v.get('metric', 'test_model-accuracy_no_oov')
        plot_items(ax, plot_v['x_axis'], plot_v['viz'], title=plot_k, y_label=plot_v.get('y_label', 'Test UA (%)'),
                   colors=plot_v.get('colors', None), agg_fn=agg_fn,
                   metric=metrics)
    max_y = round(max([ax.get_ylim()[1] for ax in axs]))
    min_y = 0
    stepsize = 1
    if axis_lim is not None and 'y' in axis_lim:
        min_y, max_y = axis_lim['y']
        stepsize = axis_lim['step'] if 'step' in axis_lim else 1
    for ai, ax in enumerate(axs):
        if axis_lim is not None and hasattr(axis_lim, '__iter__'):
            min_y, max_y = axis_lim[ai]['y']
            stepsize = axis_lim[ai]['step'] if 'step' in axis_lim[ai] else 1
        ax.set_ylim([min_y, max_y])
        ax.set_yticks(np.arange(min_y, max_y, stepsize))
        # print(ax.get_legend_handles_labels())
    # plt.savefig('/Users/robert/Desktop/test.svg', format='svg', dpi=300)
    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])
    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
    plt.show()
    return fig, axs


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

    side_by_side({
        '': {
            'x_axis': 'epoch',
            # 'metric': 'test_model-sparse_categorical_accuracy',
            'metric': 'test_model-accuracy_no_oov',
            'viz': {
                'DIPLE-4': 'DiPLeAgent_200A_70E_50B_sparse(directed-3)_05-02-2024_00_25',
                'DIPLE-2': 'DiPLeAgent_100A_30E_50B_sparse(directed-3)_05-02-2024_05_25',
                'DIPLE-21': 'DiPLeAgent_100A_20E_50B_sparse(directed-3)_05-02-2024_04_40',
                'L2C': 'L2CAgent_200A_24E_50B_sparse(directed-3)_05-02-2024_07_10',
            },
        },
    })
    """
    side_by_side({
        '0.001 LR': {
            'x_axis': 'epoch',
            'metric': 'test_model-sparse_categorical_accuracy',
            'viz': {
                'BN': 'P2PAgent_50A_100E_32B_sparse(directed-3)_13-10-2022_17_20',
                'BN-Mom': 'P2PAgent_50A_100E_32B_sparse(directed-3)_13-10-2022_23_09',
                'BN-NoES': 'P2PAgent_50A_100E_32B_sparse(directed-3)_13-10-2022_20_27',
                'No-BN': 'P2PAgent_50A_100E_32B_sparse(directed-3)_13-10-2022_14_48',
            },
        },
    })
    """
    #   x_axises=['epoch'], # 'round', 'examples', 'epoch', 'comms', 'acomms'
