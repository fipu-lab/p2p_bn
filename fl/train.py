from common import set_seed
from data.reddit import clients_data
from fl.federated_averaging import train_fed_avg


def do_train(num_clients, num_train_clients, client_pars, server_pars, model_v,
             batch_size=50, epochs=10, client_weighting='num_examples',
             round_num=None, seed=None, starting_client=0, accuracy_step='epoch'):
    train, val, test = clients_data.load_clients_data(num_clients, starting_client)
    set_seed(seed)
    train_fed_avg(train, val, test, num_train_clients,
                  batch_size, epochs, client_pars, server_pars,
                  model_v, client_weighting, round_num, accuracy_step)

