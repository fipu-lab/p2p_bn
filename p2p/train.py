from p2p.p2p_training import train_loop
from data import clients_data


def do_train(agent_class, num_clients=50, batch_size=50,
             model_pars={"model_v": 4, "lr": 0.005, "default_weights": True},
             agent_pars=None,
             graph_pars={'graph_type': 'sparse', 'num_neighbors': 2, 'directed': True, 'time_varying': -1},
             epochs=40, seed=None, starting_client=0, accuracy_step='epoch'
             ):
    train, val, test = clients_data.load_clients_data(num_clients, starting_client)
    train_loop(
        agent_class=agent_class, train=train, val=val, test=test,
        batch_size=batch_size, model_pars=model_pars,
        graph_pars=graph_pars, agent_pars=agent_pars, epochs=epochs,
        seed=seed, accuracy_step=accuracy_step)
