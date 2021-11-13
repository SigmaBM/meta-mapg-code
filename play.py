import argparse
import json
from misc.torch_utils import change_name
import os
import os.path as osp
import random

import numpy as np
import torch
from numpy.core.fromnumeric import argpartition
from torch.cuda import memory

from gym_env import make_env
from gym_env.mujoco.src.multiagent_mujoco import MujocoMulti
from meta.meta_agent import MetaAgent
from meta.peer import Peer
from misc.rl_utils import collect_trajectory
from misc.utils import set_log
from misc.video_recorder import VideoRecorder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    param_path = osp.join(args.log_path, 'params.json')
    model_path = osp.join(args.log_path, 'models', str(args.version))

    with open(param_path, 'r') as f:
        param = json.load(f)
    dict_to_args(args, param)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)
    
    env = make_env(args)
    env.seed(args.seed)
    if args.record:
        env_args = {
            "scenario": args.env_name,
            "agent_conf": args.agent_conf,
            "agent_obsk": args.agent_obsk,
            "episode_limit": args.ep_horizon,
            "inv_rew": False,
        }
        rec_env = VideoRecorder(
            MujocoMulti(env_args=env_args),
            "videos",
            lambda x: x % args.ep_horizon == 0 and x < (args.chain_horizon + 1) * args.ep_horizon,
            args.ep_horizon
        )
        rec_env.seed(args.seed)
        rec_env.render()

    # Set a temporal log path
    args.log_path = "./log_for_test"
    if not os.path.exists(args.log_path + "/log"):
        os.makedirs(args.log_path + "/log")
    log = set_log(args)

    meta_agent = MetaAgent(log, None, args, name="meta-agent", i_agent=0)
    peer = Peer(log, None, args, name="peer", i_agent=1)

    print(f"Load meta-agent's MODEL from {model_path}")
    meta_agent = torch.load(model_path)
    print(f"Load peer's ACTOR from {args.peer_path}")
    actor = torch.load(args.peer_path)
    actor = change_name(actor, old="teammate", new="peer")
    peer.actor.load_state_dict(actor, False)

    agents = [meta_agent, peer]
    actors = [agent.actor for agent in agents]

    for i_joint in range(args.chain_horizon + 1):
        memory, scores = collect_trajectory(agents, actors, env, args)
        print(f"Joint policy {i_joint} - reward 0: {scores[0]}, reward 1: {scores[1]}")

        if args.record:
            collect_trajectory(agents, actors, rec_env, args)

        phis = []
        for agent, actor in zip(agents, actors):
            phi = agent.inner_update(actor, memory, i_joint, is_train=False)
            phis.append(phi)
        
        actors = phis


def dict_to_args(args, param):
    for k, v in param.items():
        args.__dict__[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--peer-path", type=str, default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    main(args)
