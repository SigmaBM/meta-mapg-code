import numpy as np
from multiprocessing import Process, Pipe
from copy import deepcopy


def worker(remote, parent_remote, env_fn_wrapper):
    """Worker class

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, dones, info = env.step(data)
            remote.send((ob, reward, dones, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'seed':
            env.seed(data)
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError


class VecEnv(object):
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        actions = np.stack(actions, axis=1)
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    """Vectorized environment class that collects samples in parallel using subprocesses

    Args:
        env_fns (list): list of gym environments to run in subprocesses

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, env_fns):
        self.env = env_fns[0]()
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), self.observation_space, self.action_space)

    def seed(self, value):
        for i_remote, remote in enumerate(self.remotes):
            remote.send(('seed', value + i_remote))

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        self.remotes[0].send(('render', None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def sample_personas(self, is_train, is_val=True, path="./"):
        return self.env.sample_personas(is_train=is_train, is_val=is_val, path=path)

    def __len__(self):
        return self.nenvs


class DummyVecMAEnv(VecEnv):
    """Vectorized multi-agent environment class that collects samples sequentially

    Args:
        env_fns (list): list of gym environments to run sequentially

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_agents = env.n_agents
        # List/tuple of each agent's observation space
        self.observation_space = env.observation_space
        # List/tuple of each agent's action space
        self.action_space = env.action_space
        self.state_shape = env.get_env_info()["state_shape"]
        self.observation_shape = [ob.shape[0] for ob in self.observation_space]
        self.action_shape = [ac.shape[0] for ac in self.action_space]
        VecEnv.__init__(self, len(env_fns), self.observation_space, self.action_space)

        self.buf_state = np.zeros((self.num_envs, self.state_shape), dtype=np.float32)
        self.buf_obs = [np.zeros((self.num_envs, self.observation_shape[i]), dtype=np.float32) for i in range(self.num_agents)]
        self.buf_rews = [np.zeros((self.num_envs,), dtype=np.float32) for i in range(self.num_agents)]
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None     # [np.ndarray(self.num_envs, self.action_shape[i]) for i in range(self.num_agents)]
    
    def seed(self, value):
        for e in range(self.num_envs):
            self.envs[e].seed(value + e)

    def step(self, actions):
        # actions: List[np.ndarray[nenv, ac_shape]]]
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        assert len(actions) == self.num_agents, "wrong number of agents, should be {}".format(self.num_agents)
        self.actions = actions

    def step_wait(self):
        for e in range(self.num_envs):
            action = [self.actions[i][e] for i in range(self.num_agents)]
            rew, self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)

            if len(rew) == 1:
                # shared reward
                rew = [rew for _ in range(self.num_agents)]
            self.buf_state[e] = self.envs[e].get_state()
            obs = self.envs[e].get_obs()
            for i in range(self.num_agents):
                self.buf_rews[i][e] = rew[i]
                self.buf_obs[i][e] = obs[i]
        return deepcopy(self.buf_obs), deepcopy(self.buf_rews), deepcopy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self):
        for e in range(self.num_envs):
            self.buf_state[e] = self.envs[e].reset()
            obs = self.envs[e].get_obs()
            for i in range(self.num_agents):
                self.buf_obs[i][e] = obs[i]
        return deepcopy(self.buf_obs)

    def render(self):
        self.envs[0].render(mode="human")

    def close(self):
        for e in range(self.num_envs):
            self.envs[e].close()

    def __len__(self):
        return self.nenvs