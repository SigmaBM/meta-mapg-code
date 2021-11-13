import gym
from gym.envs.registration import register
from .mujoco.src.multiagent_mujoco import MujocoMulti
from .multiprocessing_env import DummyVecMAEnv
from .wrapper import Wrapper


register(
    id='IPD-v0',
    entry_point='gym_env.ipd.ipd_env:IPDEnv',
    kwargs={'args': None},
    max_episode_steps=150
)

register(
    id='RPS-v0',
    entry_point='gym_env.rps.rps_env:RPSEnv',
    kwargs={'args': None},
    max_episode_steps=150
)

def make_each_env(args):
    def thunk():
        env_args = {
            "scenario": args.env_name,
            "agent_conf": args.agent_conf,
            "agent_obsk": args.agent_obsk,
            "episode_limit": args.ep_horizon,
            "inv_rew": False,
        }
        env = Wrapper(MujocoMulti(env_args=env_args), args)
        return env
    return thunk


def make_env(args):
    if args.env_name in ['IPD-v0', 'RPS-v0']:
        env = gym.make(args.env_name, args=args)
        env._max_episode_steps = args.ep_horizon
    else:
        # multi-agent mujoco environment
        env = [make_each_env(args) for _ in range(args.traj_batch_size)]
        env = DummyVecMAEnv(env)
        # TODO: normalized vec env (maybe)
    return env
