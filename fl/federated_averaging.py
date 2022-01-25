from common import *
import environ
from datetime import datetime

if environ.is_collab():
    import nest_asyncio
    nest_asyncio.apply()

import time
import tensorflow_federated as tff
from tensorflow_federated.python.learning import ClientWeighting


DATA_SPEC = 1
MODEL_VERSION = 1
ACC_HIST = {}


def create_keras_model_fed():
    return create_model(MODEL_VERSION, do_compile=False)


def create_model_fed():
    km = create_keras_model_fed()
    return tff.learning.from_keras_model(km,
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                         input_spec=DATA_SPEC)


def create_tf_dataset_for_client(c, batch_size=15):
    x, y = np.array(c[0]), np.array(c[1])
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(batch_size).batch(batch_size)


def get_sample_clients(c_list, num_clients):
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]


def state_to_model(s_state):
    model = create_keras_model_fed()
    s_state.model.assign_weights_to(model)
    compile_model(model)
    return model


def avg_model_acc(model, dataset, cl_keys, desc, batch_size):
    # reset_compiled_metrics(model)
    accs = []
    # all_metrics = {}
    for key in tqdm(cl_keys, desc="Evaluating {}".format(desc), position=0, leave=False):
        if len(dataset[key][1]) < 1:
            continue
        metrics = model.evaluate(dataset[key][0], dataset[key][1], batch_size=batch_size, verbose=0)
        accs.append(metrics[1])
        """
        metrics = eval_model_metrics(model, create_tf_dataset_for_client(dataset[key], batch_size))
        for mk, mv in metrics.items():
            if mk not in all_metrics:
                all_metrics[mk] = []
            all_metrics[mk].append(mv)
        accs.append(metrics['accuracy_no_oov'])
        """
    """
    for m_name, m_val in all_metrics.items():
        print("\t{}\t{:.3%}\t{:.3%}".format(m_name, np.average(np.array(m_val)), np.median(np.array(m_val))))
    """
    ACC_HIST[list(ACC_HIST.keys())[-1]][desc] = accs
    accs = np.array(accs)
    return np.average(accs), np.median(accs)


def print_avg_model_acc(model, dataset_dict, cl_keys, batch_size):
    for key in dataset_dict.keys():
        # avg_model_acc(model, dataset_dict[key], cl_keys, key, batch_size)
        print("\t{}\t{:.3%}\t{:.3%}\n".format(key, *avg_model_acc(model, dataset_dict[key], cl_keys, key, batch_size)), flush=True, end='')


def print_accuracy(server_state, train_clients, val_clients, test_clients, client_inds,
                   round_num=1, epoch=1, round_examples=0, total_examples=0, total_client_samples=0, batch_size=50):
    fed_model = state_to_model(server_state)
    msg = "Round: {}\tEpoch: {}\n".format(round_num, epoch) + \
          "\tExamples: ({}/{})\n".format(round_examples, total_examples) + \
          "\tAgents\tMean\tMedian"
    print(msg, flush=True)
    ACC_HIST[round_num] = {"Epoch": epoch, "Examples": total_examples, "Sampled_Clients": total_client_samples}
    print_avg_model_acc(fed_model, {"Train": train_clients, "Valid": val_clients, "Test": test_clients}, client_inds, batch_size)


def train_fed_avg(train_clients,
                  val_clients,
                  test_clients,
                  num_train_clients,
                  batch_size=15,
                  epochs=5,
                  client_pars=None,
                  server_pars=None,
                  model_v=1,
                  client_weighting=None,
                  checkpoint_round=-1,
                  accuracy_step='epoch'):

    start_time = time.time()

    global MODEL_VERSION, ACC_HIST
    MODEL_VERSION = model_v
    ACC_HIST = {}

    client_default = {"lr": 0.005, "decay": 0}
    client_pars = client_default if client_pars is None else {**client_default, **client_pars}
    server_default = {"lr": 0.005, "decay": 0}
    server_pars = server_default if server_pars is None else {**server_default, **server_pars}

    client_inds = list(range(len(train_clients)))
    clients_num = len(client_inds)
    examples = sum([len(train_clients[ind][1]) for ind in range(clients_num)])
    max_examples = examples * epochs
    total_examples, total_client_samples = 0, 0
    client_weighting = client_weighting or "num_examples"

    msg = "FL training (V={} {}), Num clients per round: {}, batch size: {}, epochs: {}, examples: {}, server_pars: {}, client_pars: {}" \
        .format(model_v, client_weighting, num_train_clients, batch_size, epochs, examples, server_pars, client_pars)
    print(msg, flush=True)

    global DATA_SPEC
    DATA_SPEC = create_tf_dataset_for_client(train_clients[0], batch_size).element_spec

    iterative_process = tff.learning.federated_averaging.build_federated_averaging_process(
        model_fn=create_model_fed,
        server_optimizer_fn=lambda: Adam(learning_rate=server_pars['lr']),
        client_optimizer_fn=lambda: Adam(learning_rate=client_pars['lr']),
        client_weighting=ClientWeighting.UNIFORM if client_weighting == 'uniform' else ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=True
    )

    server_state = iterative_process.initialize()

    ckpt_manager = tff.simulation.FileCheckpointManager(
        root_dir='models/fl_ckpts/{}V_{}_{}TR_{}CL_{}B_{}E/'.format(model_v, client_weighting, clients_num, num_train_clients, batch_size, epochs),
        prefix='ckpt_',
        keep_total=999999999,
        keep_first=True
    )

    round_num = 0
    if type(checkpoint_round) == str:
        server_state = ckpt_manager.load_checkpoint(server_state, checkpoint_round)
        print_accuracy(server_state, train_clients, val_clients, test_clients, client_inds, checkpoint_round, batch_size=batch_size)
        return
    if type(checkpoint_round) == int and checkpoint_round > 0:
        round_num = checkpoint_round
        total_examples += int(((examples / clients_num) * num_train_clients) * round_num)
        print("Loading checkpoint:", checkpoint_round)
        server_state = ckpt_manager.load_checkpoint(server_state, checkpoint_round)

    """
    if 'epoch' in accuracy_step:
        epochs_num = accuracy_step.replace('epoch', '').strip() or 1
        accuracy_step = int(examples / ((examples / clients_num) * num_train_clients)) * epochs_num
    elif 'round' in accuracy_step:
        accuracy_step = int(accuracy_step.replace('round', '').strip() or 1)
    elif accuracy_step == 'never':
        accuracy_step = 9999999999
    """

    pbar = tqdm(total=examples, position=0, leave=False)
    while total_examples < max_examples:
        round_num += 1
        # Sample train clients to create a train dataset
        if len(client_inds) <= num_train_clients:
            train_clients_ds = client_inds
        else:
            train_clients_ds = get_sample_clients(client_inds, num_train_clients)
        # print('\nSampling {} new clients.'.format(len(train_clients_ds)), flush=True)

        total_client_samples += len(train_clients_ds)
        train_datasets = [create_tf_dataset_for_client(train_clients[c], batch_size) for c in train_clients_ds]

        # Apply federated training round
        server_state, server_metrics = iterative_process.next(server_state, train_datasets)
        if checkpoint_round is not None:
            ckpt_manager.save_checkpoint(server_state, round_num=round_num)

        round_examples = server_metrics['stat']['num_examples']
        pbar.set_postfix({**memory_info(), **{"TP": time_elapsed_info(start_time)}})
        total_examples += round_examples

        pbar.update(round_examples)
        if pbar.n >= pbar.total:
            diff = pbar.n - pbar.total
            pbar.close()
            epoch = round(total_examples / examples)
            print_accuracy(server_state, train_clients, val_clients, test_clients, client_inds,
                           round_num, epoch, round_examples, total_examples, total_client_samples, batch_size)

            pbar = tqdm(total=examples, position=0, leave=False)
            pbar.update(diff)
    pbar.close()
    print("Train time: {}".format(time_elapsed_info(start_time)), flush=True)

    save_json('log/fl_{}C_{}TR_{}V({}S-{}C)_{}E_{}_{}.json'.format(clients_num,
                                                                   num_train_clients, model_v,
                                                                   str(server_pars['lr']).replace('.', '_'),
                                                                   str(client_pars['lr']).replace('.', '_'),
                                                                   epochs, client_weighting,
                                                                   datetime.now().strftime("%d-%m-%Y_%H_%M")), ACC_HIST)



