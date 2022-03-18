import os
import dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)


# def save_env_vars():
#     dotenv.set_key(dotenv_file, "AGENT_COUNTER", os.environ["AGENT_COUNTER"])


def is_collab():
    return not ('USER' in os.environ and os.environ['USER'] == 'robert')


def next_agent_id():
    counter = int(os.environ.get("AGENT_COUNTER", 0))
    os.environ['AGENT_COUNTER'] = str(counter + 1)
    return counter


def set_agent_id(agent_id):
    os.environ['AGENT_COUNTER'] = str(agent_id)


def get_devices():
    devices = os.environ.get('DEVICES', '').split(',')
    return [d.strip() for d in devices if len(d.strip()) > 0]


def set_visible_devices(devices):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Visible devices already set
        return
    devices = devices or 'CPU'
    devices = str.upper(devices)

    vis_dev = []
    for dev in devices.split(','):
        if 'CPU' in dev:
            viz_ind = '-1'
        elif 'GPU' in dev:
            viz_ind = dev.replace('GPU', '').replace(':', '').strip()
        else:
            raise Exception('Unsupported device')

        if viz_ind in vis_dev:
            raise Exception('Device already added')
        vis_dev.append(viz_ind)

    viz_dev = sorted(vis_dev)

    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(viz_dev).replace('-1, ', '')
    devices = []
    if '-1' in viz_dev:
        devices.append('CPU')
        viz_dev.remove('-1')

    for dev_i in range(len(viz_dev)):
        devices.append('GPU:{}'.format(dev_i))

    os.environ['DEVICES'] = ', '.join(devices)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def check_devices():
    set_visible_devices(os.environ.get('DEVICES', ''))
