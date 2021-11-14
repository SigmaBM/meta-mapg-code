#!/usr/bin/env python3

import argparse
import datetime
import os
import os.path as osp
import pprint
import sys

sys.path.append(osp.abspath(osp.join(__file__, "../..")))

import numpy as np
import torch
from gym_env.mujoco.src.multiagent_mujoco import MujocoMulti
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb as ActorProb0
from tianshou.utils.net.continuous import Critic as Critic0
from torch.utils.tensorboard import SummaryWriter

from pretrain_mujoco.macollector import MACollector
from pretrain_mujoco.offpolicy import offpolicy_trainer
from pretrain_mujoco.utils import ActorProb, BasicLogger, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HalfCheetah-v2')
    parser.add_argument('--agent-conf', type=str, default='2x3')
    parser.add_argument('--agent-obsk', type=int, default=None)
    parser.add_argument('--ep-limit', type=int, default=200)
    parser.add_argument('--inv-rew',action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, default=64)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=int, default=0.1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='debug')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-interval', type=int, default=0)
    return parser.parse_args()


def main(args=get_args()):
    env_args = {
            "scenario": args.task,
            "agent_conf": args.agent_conf,
            "agent_obsk": args.agent_obsk,
            "episode_limit": args.ep_limit,
            "inv_rew": args.inv_rew,
        }
    env = MujocoMulti(env_args=env_args)
    args.state_shape = [ob.shape or ob.n for ob in env.observation_space]
    args.action_shape = [ac.shape or ac.n for ac in env.action_space]
    args.max_action = max([ac.high[0] for ac in env.action_space])
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    if args.training_num > 1:
        # train_envs = SubprocVectorEnv(
        #     [lambda: MujocoMulti(env_args=env_args) for _ in range(args.training_num)])
        train_envs = DummyVectorEnv(
            [lambda: MujocoMulti(env_args=env_args) for _ in range(args.training_num)])
    else:
        train_envs = MujocoMulti(env_args=env_args)
    # test_envs = gym.make(args.task)
    # test_envs = SubprocVectorEnv(
    #     [lambda: MujocoMulti(env_args=env_args) for _ in range(args.test_num)])
    test_envs = DummyVectorEnv(
        [lambda: MujocoMulti(env_args=env_args) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policies = []
    for i in range(env.n_agents):
        actor = ActorProb(
            "teammate" + str(i), args.state_shape[i], args.action_shape[i], args.hidden_sizes,
            args.max_action, args.device, True, False).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1 = Critic(
            "teammate" + str(i), args.state_shape[i], args.action_shape[i], args.hidden_sizes,
            args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        critic2 = Critic(
            "teammate" + str(i), args.state_shape[i], args.action_shape[i], args.hidden_sizes,
            args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        if args.auto_alpha:
            target_entropy = -np.prod(env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)
        
        policies.append(SACPolicy(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, 
            args.tau, args.gamma, args.alpha, estimation_step=args.n_step,
            action_space=env.action_space[i]
        ))

    # load a previous policy
    if args.resume_path:
        raise NotImplementedError

    # collector
    if args.training_num > 1:
        buffers = [VectorReplayBuffer(args.buffer_size, len(train_envs)) for _ in range(env.n_agents)]
    else:
        buffers = [ReplayBuffer(args.buffer_size) for _ in range(env.num_agents)]
    train_collector = MACollector(policies, train_envs, buffers, exploration_noise=True)
    test_collector = MACollector(policies, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_sac'
    log_path = os.path.join(args.logdir, 'sac', args.task, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policies, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, args.test_num,
            args.batch_size, logger=logger, update_per_step=args.update_per_step, 
            test_in_train=False, save_interval=args.save_interval)
        pprint.pprint(result)

    # Let's watch its performance!
    [pi.eval() for pi in policies]
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean(axis=0)}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    main()
