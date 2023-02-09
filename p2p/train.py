from p2p.p2p_training import train_loop


def do_train(agent_pars,
             agent_data_pars,
             model_pars,
             graph_pars,
             sim_pars
             ):
    if isinstance(agent_pars, dict):
        agent_pars = [agent_pars]
    for ap in agent_pars:
        assert all(k in ap for k in ["agent_class"])
    if isinstance(agent_data_pars, dict):
        agent_data_pars = [agent_data_pars]
    for adp in agent_data_pars:
        assert all(k in adp for k in ["agents_data", "num_clients", "batch_size"])
    if isinstance(model_pars, dict):
        model_pars = [model_pars]
    for mp in model_pars:
        assert all(k in mp for k in ["model_mod"])
    assert all(k in graph_pars for k in ["graph_type", "num_neighbors", "directed"])

    train_loop(agent_pars=agent_pars,
               agent_data_pars=agent_data_pars,
               model_pars=model_pars,
               graph_pars=graph_pars,
               sim_pars=sim_pars)
