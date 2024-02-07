import os
import json

import psutil
import GPUtil

import random
import tensorflow as tf
import numpy as np
import time


def set_seed(seed):
    if seed:
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # print("Seed:", seed)


def available_device_memory(device_name):
    if 'GPU' in device_name.upper():
        gpu_cur, gpu_total = gpu_memory(device_name, 'MB')
        return gpu_total - gpu_cur
    elif 'CPU' in device_name.upper():
        return psutil.virtual_memory()[1] / 1024**2


def total_device_memory(device_name):
    if 'GPU' in device_name.upper():
        gpu_cur, gpu_total = gpu_memory(device_name, 'MB')
        return gpu_total
    elif 'CPU' in device_name.upper():
        return psutil.virtual_memory()[0] / 1024**2


def gpu_memory(gpu_name, units='MB'):
    power_f = ['B', 'MB', 'GB'].index(units) + 1
    viz_devs = os.environ["CUDA_VISIBLE_DEVICES"].split(', ')
    gpu_ind = gpu_name.replace('GPU', '').replace(':', '').strip()

    usage = tf.config.experimental.get_memory_info(gpu_name)
    current = round(usage['current'] / (1024 ** power_f), 2)
    total = round(GPUtil.getGPUs()[int(viz_devs[int(gpu_ind)])].memoryTotal * 1024 ** 2 / (1024 ** power_f), 2)
    return current, total


def memory_info():
    gpu_devices = tf.config.list_physical_devices('GPU')
    mem_dict = {}
    for device in gpu_devices:
        dev_name = device.name.replace('/physical_device:', '')
        gpu_cur, gpu_total = gpu_memory(dev_name, 'GB')
        mem_dict[dev_name] = "{}/{} GB".format(gpu_cur, gpu_total)
    ram_mem = psutil.virtual_memory()
    mem_dict['RAM'] = "{}/{} GB".format(round((ram_mem[0]-ram_mem[1]) / 1024**3, 2), round(ram_mem[0] / 1024**3, 2))
    return mem_dict


def time_elapsed_info(start_time):
    minutes = round((time.time() - start_time)/60)
    hours = int(minutes / 60)
    minutes = minutes % 60
    return "{:02d}:{:02d}h".format(hours, minutes)


def save_json(filename, json_dict):

    class NumpyValuesEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    num = ''

    def parsed_name():
        return filename.split('.json')[0] + num + '.json'
    while os.path.exists(parsed_name()):
        if num == '':
            num = '_(1)'
        else:
            num = '_({})'.format(int(num[num.index('(')+1:-1]) + 1)
    with open(parsed_name(), "w") as outfile:
        json.dump(json_dict, outfile, indent=None, cls=NumpyValuesEncoder)
    print("Saved to", parsed_name())
    return parsed_name()
