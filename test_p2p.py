import argparse
import json
import importlib

from p2p.train import do_train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent",
                        default='P2PAgent',
                        type=str,
                        required=True,
                        help="Agent class to be used in simulations")
    parser.add_argument("--clients",
                        required=True,
                        type=int,
                        help="Number of clients to be used in simulation")
    parser.add_argument("--batch_size",
                        default=50,
                        type=int,
                        help="Batch size (default 50)")
    parser.add_argument("--epochs",
                        default=30,
                        type=int,
                        help="Number of cumulative epoch to train (default 30)")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed (default None)")
    parser.add_argument("--model_v",
                        default=4,
                        type=int,
                        help="Model version to be used in simulation (default 4)")
    parser.add_argument("--lr",
                        default=0.005,
                        type=float,
                        help="Agent learning rate (default 0.005)")
    parser.add_argument("--agent_pars",
                        default=None,
                        type=str,
                        help="Json-type string with custom agent parameters (default None)")
    parser.add_argument("--graph_type",
                        default='sparse',
                        type=str,
                        help="Graph type to create as a communication base (default sparse)")
    parser.add_argument("--neighbors",
                        default=3,
                        type=int,
                        help="Number of neighbors each agent has (default 3)")
    parser.add_argument("--directed",
                        action='store_true',
                        help="Set this flag for directed communication (default false)")
    parser.add_argument("--vary",
                        default=-1,
                        type=int,
                        help="Time-varying interval of changing communication matrix (default -1)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # mod = __import__('p2p').agents
    # if not hasattr(mod, args.agent):
    #     raise ValueError(f"{args.agent} not found in 'p2p.agents' module")

    do_train(importlib.import_module('p2p.agents.' + args.agent),
             importlib.import_module('data.reddit.clients_data'),
             num_clients=args.clients,
             batch_size=args.batch_size,
             model_pars={"model_v": args.model_v, "lr": args.lr, "default_weights": True},
             agent_pars=None if args.agent_pars is None else json.loads(args.agent_pars),
             graph_pars={'graph_type': args.graph_type,
                         'num_neighbors': args.neighbors,
                         'directed': args.directed,
                         'time_varying': args.vary},
             epochs=args.epochs,
             seed=args.seed,
             accuracy_step='epoch')
