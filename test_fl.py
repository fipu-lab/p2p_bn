import argparse
from fl import do_train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clients",
                        required=True,
                        type=int,
                        help="Number of clients to be used in simulation")
    parser.add_argument("--sample_size",
                        required=True,
                        type=int,
                        help="Number of clients to sample in each FL round")
    parser.add_argument("--epochs",
                        default=30,
                        type=int,
                        help="Number of cumulative epoch to train (default 30)")
    parser.add_argument("--batch_size",
                        default=50,
                        type=int,
                        help="Batch size (default 50)")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed (default None)")
    parser.add_argument("--model_v",
                        default=2,
                        type=int,
                        help="Model version to be used in simulation (default 2)")
    parser.add_argument("--slr",
                        default=0.005,
                        type=float,
                        help="Server learning rate (default 0.005)")
    parser.add_argument("--clr",
                        default=0.005,
                        type=float,
                        help="Client learning rate (default 0.005)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    do_train(num_clients=args.clients,
             num_train_clients=args.sample_size,
             batch_size=args.batch_size,
             epochs=args.epochs,
             client_pars={"lr": args.slr, "decay": 0},
             server_pars={"lr": args.clr, "decay": 0},
             model_v=args.model_v,
             client_weighting='num_examples',
             round_num=None,
             seed=args.seed,
             accuracy_step='epoch'
             )
