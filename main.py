import argparse
import datetime
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from meta.meta_agent import MetaAgent
from misc.utils import load_config, set_log
from tester import meta_test
from trainer import meta_train
from validator import meta_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Set logging
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.log_path = osp.join(args.log_path, "meta-mapg", args.env_name,
                             f'seed_{args.seed}_{t0}-{args.env_name.replace("-", "_")}_metamapg')
    if not os.path.exists(args.log_path + "/log"):
        os.makedirs(args.log_path + "/log")

    log = set_log(args)
    # tb_writer = SummaryWriter(args.log_path + '/log/tb_{0}'.format(args.log_name))
    tb_writer = None    # SummaryWriter cannot be serialized!

    # Use a JSON file to save arguments
    with open(osp.join(args.log_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)

    # For GPU, Set start method for multithreading
    if device == torch.device("cuda"):
        # torch.multiprocessing.set_start_method('spawn')
        mp = torch.multiprocessing.get_context("spawn")
    
    # Split training, validation and test pretrain model randomly
    if args.env_name not in ["IPD-v0", "RPS-v0"]:
        split_pretrained_models(args)

    # Initialize shared meta-agent
    shared_meta_agent = MetaAgent(log, tb_writer, args, name="meta-agent", i_agent=0)
    shared_meta_agent.share_memory()

    if args.resume_path:
        log[log.name].info("Resume training from {}".format(args.resume_path))

    # Used for debug
    # meta_train(shared_meta_agent, mp.Manager().dict(), 0, log, args)

    # Begin either meta-train or meta-test
    if not args.test_mode:
        # Start meta-train
        processes, process_dict = [], mp.Manager().dict()
        for rank in range(args.n_process):
            p = mp.Process(
                target=meta_train,
                args=(shared_meta_agent, process_dict, rank, log, args))
            p.start()
            processes.append(p)
            time.sleep(0.1)

        p = mp.Process(
            target=meta_val,
            args=(shared_meta_agent, process_dict, -1, log, args))
        p.start()
        processes.append(p)
        time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()
    else:
        # Start meta-test
        meta_test(shared_meta_agent, log, tb_writer, args)


def split_pretrained_models(
    args,
    range_tr_va_lb: int = 1,
    range_tr_va_ub: int = 300,
    range_ts_lb: int = 475,
    range_ts_ub: int = 500,
    num_tr: int = 275
) -> None:
    model_path = args.model_path if args.model_path is not None else osp.join(".", "pretrain_model", args.env_name)
    list_all = os.listdir(model_path)
    # str2int -> sort -> int2str
    list_all = list(map(int, list_all))
    list_all.sort()
    list_all = list(map(str, list_all))
    rand_list_tr_va = np.random.permutation(list_all[range_tr_va_lb - 1: range_tr_va_ub])
    args.list_tr = rand_list_tr_va[: num_tr]
    args.list_va = rand_list_tr_va[num_tr: ]
    args.list_ts = list_all[range_ts_lb - 1: range_ts_ub]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="meta-mapg")

    # Algorithm
    parser.add_argument(
        "--opponent-shaping", action="store_true",
        help="If True, include opponent shaping in meta-optimization")
    parser.add_argument(
        "--traj-batch-size", type=int, default=64,
        help="Number of trajectories for each inner-loop update (K Hyperparameter)")
    parser.add_argument(
        "--n-process", type=int, default=5,
        help="Number of parallel processes for meta-optimization")
    parser.add_argument(
        "--actor-lr-inner", type=float, default=0.1,
        help="Learning rate for actor (inner loop)")
    parser.add_argument(
        "--actor-lr-outer", type=float, default=0.0001,
        help="Learning rate for actor (outer loop)")
    parser.add_argument(
        "--value-lr", type=float, default=0.00015,
        help="Learning rate for value (outer loop)")
    parser.add_argument(
        "--entropy-weight", type=float, default=0.5,
        help="Entropy weight in the meta-optimization")
    parser.add_argument(
        "--discount", type=float, default=0.96,
        help="Discount factor in reinforcement learning")
    parser.add_argument(
        "--lambda_", type=float, default=0.95,
        help="Lambda factor in GAE computation")
    parser.add_argument(
        "--chain-horizon", type=int, default=5,
        help="Markov chain terminates when chain horizon is reached")
    parser.add_argument(
        "--n-hidden", type=int, default=64,
        help="Number of neurons for hidden network")
    parser.add_argument(
        "--max-grad-clip", type=float, default=10.0,
        help="Max norm gradient clipping value in meta-optimization")
    parser.add_argument(
        "--test-mode", action="store_true",
        help="If True, perform meta-test instead of meta-train")
    parser.add_argument(
        "--max-train-iteration", type=int, default=1e5,
        help="Terminate program when max train iteration is reached")
    parser.add_argument(
        "--use-rnn", action="store_true", 
        help="Whether to use LSTM model in meta-agent"
    )
    parser.add_argument(
        "--resume-path", type=str,
        help="Load a trained meta-agent to resume training"
    )

    # Env
    parser.add_argument(
        "--env-name", type=str, default="",
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-horizon", type=int, default=150,
        help="Episode is terminated when max timestep is reached")
    parser.add_argument(
        "--n-agent", type=int, default=2,
        help="Number of agents in a shared environment")
    # MAMujoco
    parser.add_argument("--agent-conf", type=str, default="2x3")
    parser.add_argument("--agent-obsk", type=int, default=None)

    # Misc
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--log-path", type=str, default=".",
        help="Path to save log files"
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path where pretrained models save"
    )
    parser.add_argument(
        "--save-interval", type=int, default=0,
        help="Frequency of saving model"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config that replaces default params with experiment specific params")

    args = parser.parse_args()

    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    # args.log_name = \
    #     "env::%s_seed::%s_opponent_shaping::%s_traj_batch_size::%s_chain_horizon::%s_" \
    #     "actor_lr_inner::%s_actor_lr_outer::%s_value_lr::%s_entropy_weight::%s_" \
    #     "max_grad_clip::%s_prefix::%s_log" % (
    #         args.env_name, args.seed, args.opponent_shaping, args.traj_batch_size, args.chain_horizon,
    #         args.actor_lr_inner, args.actor_lr_outer, args.value_lr, args.entropy_weight,
    #         args.max_grad_clip, args.prefix)
    # Too long!
    args.log_name = "log"

    main(args=args)
