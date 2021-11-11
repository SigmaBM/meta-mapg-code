import argparse
import os.path as osp
import sys

from torch import nn

sys.path.append(osp.abspath(osp.join(__file__, "../..")))

import numpy as np
import torch
from gym_env.mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
from tianshou.policy.modelfree.sac import SACPolicy

from pretrain_mujoco.macollector import MACollector
from pretrain_mujoco.utils import ActorProb, RecurrentActorProb


def main(args):
    env_args = {
        "scenario": args.task,
        "agent_conf": args.agent_conf,
        "agent_obsk": args.agent_obsk,
        "episode_limit": args.ep_limit,
        "inv_rew": False,
    }
    env = MujocoMulti(env_args=env_args)
    args.state_shape = [ob.shape or ob.n for ob in env.observation_space]
    args.action_shape = [ac.shape or ac.n for ac in env.action_space]
    args.max_action = max([ac.high[0] for ac in env.action_space])
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    policies = []
    for i in range(env.n_agents):
        if args.rnn:
            actor = RecurrentActorProb(
                "teammate" + str(i), args.state_shape[i], args.action_shape[i], args.hidden_sizes,
                args.max_action, args.device, True, False).to(args.device)
        else:
            actor = ActorProb(
                "teammate" + str(i), args.state_shape[i], args.action_shape[i], args.hidden_sizes,
                args.max_action, args.device, True, False).to(args.device)
        
        policies.append(SACPolicy(actor, None, nn.Module(), None, nn.Module(), None, action_space=env.action_space[i]))

        if args.load_path:
            path = osp.join(args.load_path, 'models-{}'.format(i), '{}'.format(args.load_itr))
            policies[-1].actor.load_state_dict(torch.load(path))
            print("Load agent {}'s model from {}...".format(i, path))

        policies[-1].eval()
    
    collector = MACollector(policies, env)
    result = collector.collect(n_step=args.playtime, render=args.render)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HalfCheetah-v2')
    parser.add_argument('--agent-conf', type=str, default='2x3')
    parser.add_argument('--agent-obsk', type=int, default=None)
    parser.add_argument('--ep-limit', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden-sizes', type=int, default=64)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--load-itr', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rnn', action='store_true')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--playtime', type=int, default=10000)

    args = parser.parse_args()
    main(args)
