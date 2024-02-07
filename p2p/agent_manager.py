from p2p.p2p_utils import *
import time


def init_agents(agent_pars, agent_data_pars, model_pars):
    start_time = time.time()

    client_pars = {k: v for k, v in agent_data_pars.items() if k not in ['agents_data', 'batch_size', 'caching']}
    data_dict = agent_data_pars['agents_data'].load_clients_data(**client_pars)
    num_agents = len(data_dict["train"])

    # clear_def_weights_cache()

    init_on_cpu = agent_pars.get('init_on_cpu', False)
    agent_class = agent_pars['agent_class']
    pbar = tqdm(total=num_agents, position=0, leave=False, desc='Init {} agents'.format(agent_class.__name__.split('.')[-1]))
    devices = environ.get_devices()
    agents = []

    # for agent_id, (train, val, test) in enumerate(zip(train_clients, val_clients, test_clients)):
    for agent_id in range(num_agents):
        device = resolve_agent_device(agents, None, devices)
        with tf.device('CPU' if init_on_cpu else device or 'CPU'):
            clear_session()

            a_p = {k: v for k, v in agent_pars.items() if k not in ['agent_class', 'init_on_cpu']}
            a_p['data'] = {k: v[agent_id] for k, v in data_dict.items()}
            # a_p['train'], a_p['val'], a_p['test'] = train, val, test
            a_p['data_pars'], a_p['model'] = agent_data_pars, model_pars

            a = agent_class(**a_p)
            a.device = device
            a.id = agent_id
            agents.append(a)
        update_pb(pbar, agents, 1, start_time)
    pbar.close()
    print("Init {} agents: {} minutes".format(agent_class.__name__.split('.')[-1], round((time.time() - start_time) / 60)))
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
