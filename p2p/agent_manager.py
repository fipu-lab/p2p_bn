from p2p.p2p_utils import *
import time


def init_agents(agent_class, train_clients, val_clients, test_clients, batch_size, model_pars=None, agent_pars=None):
    start_time = time.time()
    model_default = {"model_v": 1, "lr": 0.001, "decay": 0, "default_weights": False}
    model_pars = model_default if model_pars is None else {**model_default, **model_pars}
    agent_pars = agent_pars or {}

    num_agents = len(train_clients)
    print("{}: {} agents, batch size: {}, model_pars: {}, agent_pars: {}".format(
        agent_class.__name__, num_agents, batch_size, model_pars, agent_pars))

    clear_def_weights_cache()

    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init agents')
    devices = environ.get_devices()
    agents = []

    for agent_id, (train, val, test) in enumerate(zip(train_clients, val_clients, test_clients)):
        device = resolve_agent_device(agents, None, devices)
        with tf.device(device or 'CPU'):
            clear_session()
            agent_pars['train'] = train
            agent_pars['val'] = val
            agent_pars['test'] = test
            agent_pars['batch_size'] = batch_size
            agent_pars['model'] = create_model(**model_pars)
            a = agent_class(**agent_pars)
            a.device = device
            a.id = agent_id
            agents.append(a)
        update_pb(pbar, agents, 1, start_time)
    pbar.close()
    print("Init agents: {} minutes".format(round((time.time() - start_time) / 60)))
    # environ.save_env_vars()
    return agents


def load_agents(agent_class, train_clients, val_clients, test_clients, batch_size, first_agent_id):
    environ.set_agent_id(first_agent_id)
    agents = []
    for train, val, test in zip(train_clients, val_clients, test_clients):
        a = agent_class(train=train,
                        val=val,
                        test=test,
                        batch_size=batch_size,
                        model=None,
                        )
        agents.append(a)
    return agents
