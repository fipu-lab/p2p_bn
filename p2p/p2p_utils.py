from models import *
import numpy as np
import tensorflow as tf


def mode(arr):
    vals, counts = np.unique([round(a) for a in arr], return_counts=True)
    return vals[counts == max(counts)][0]


def print_acc(accs, info):
    accs = np.array(accs or [0])
    if len(accs[accs > 1]):
        print("{}\t\t{}\t\t{:.4}".format(info, round(sum(accs), 2), round(np.mean(accs), 2)))
    else:
        print("{}\t\t{:.3%}\t\t{:.3%}".format(info, np.average(accs), np.median(accs)))


def calc_new_agent_metrics(agents):
    devices = environ.get_devices()
    pbar = tqdm(total=len(agents), position=0, leave=False, desc='Evaluating agents')
    for a in agents:
        device = resolve_agent_device(agents, a, devices)
        with tf.device(device or 'CPU'):
            a.calc_new_metrics()
        update_pb(pbar, agents)
    pbar.close()


def calc_agents_metrics(agents, e=0, print_metrics=None, group_by_dataset=False):
    print("Epoch:", e)
    calc_new_agent_metrics(agents)
    h = {}
    for a in agents:
        for hk, hv in a.hist.items():
            if '-' in hk:
                if group_by_dataset:
                    hk = "{}->{}".format(a.dataset_name, hk)
                if hk not in h:
                    h[hk] = []
                h[hk].append(hv[-1])

    max_len = max([len(k) for k in h.keys()]) + 1
    print(("\t{: <" + str(max_len) + "}\t\tMean\t\tMedian").format('Metric'))
    print('\t' + '-' * (max_len + 35))
    for hk, hv in h.items():
        if print_metrics is not None and hk.split('->')[-1] not in print_metrics:
            continue
        print_acc(hv, ("\t{: <" + str(max_len) + "}").format(hk + ':'))
    print('', end='', flush=True)


def resolve_agent_device(agents, agent, devices):
    if len(devices) == 0:
        return None
    if agent is None:
        if len(devices) == 1:
            return devices[0]
        dev_mem = {dev: total_device_memory(dev) * 0.9 - sum([a.memory_footprint for a in agents if a.device == dev])
                   for dev in devices}
        gpu_mem = {gpu: mem for gpu, mem in dev_mem.items() if 'GPU' in gpu}
        gpu = max(gpu_mem, key=gpu_mem.get)
        if gpu_mem[gpu] > 0:
            return gpu
        return max(dev_mem, key=dev_mem.get)
    """
    if agent.device is None:
        free_mem, mem_dev = 0, ''
        for device in devices:
            available_mem = available_device_memory(device)
            if available_mem > free_mem:
                free_mem = available_mem
                mem_dev = device

        agent.device = mem_dev
        with tf.device(agent.device):
            agent.deserialize()
        if available_device_memory(agent.device) > agent.memory_footprint * 3:
            return agent.device

        for a in agents:
            if a != agent and a.device == agent.device:
                with tf.device(a.device):
                    a.serialize()
                    a.device = None
                break
    """
    return agent.device


def new_progress_bar(total, desc=''):
    return tqdm(total=total, position=0, leave=False, desc=desc)


def update_pb(pbar, agents, n=1, start_time=None):
    pbar.update(n)
    devices = environ.get_devices()
    postfix = memory_info()
    for dev in devices:
        postfix["N_{}".format(dev)] = sum([1 for a in agents if a.device == dev])
    if start_time:
        postfix["TP"] = time_elapsed_info(start_time)
    pbar.set_postfix(postfix)


def dump_acc_hist(filename, agents, graph, info={}):
    if type(info) is dict:
        for k, v in info.items():
            if type(v) is dict:
                info[k] = {ki: str(vi) for ki, vi in v.items()}
            else:
                info[k] = str(v)
    return save_json(filename, {'agents': {a.id: a.hist for a in agents}, 'graph': graph, 'info': info})
