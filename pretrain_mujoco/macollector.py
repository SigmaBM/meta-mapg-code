import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import (
    Batch,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    CachedReplayBuffer,
    to_numpy,
)


class MACollector(object):
    """Multi-agent collector enables the policies to interact with different types of \
    envs with exact number of steps or episodes.

    :param policies: a list of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffers: a list of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.
    
    References:
        https://github.com/thu-ml/tianshou/tianshou/data/data/collector.py
    """

    def __init__(
        self,
        policies: List[BasePolicy],
        env: Union[gym.Env, BaseVectorEnv],
        buffers: Optional[List[ReplayBuffer]] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            env = DummyVectorEnv([lambda: env])
        self.env = env
        self.env_num = len(env)
        self.agt_num = env.n_agents[0]
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffers)
        self.policies = policies
        self._action_space = env.action_space
        # avoid creating attribute outside __init__
        self.reset()

    def _assign_buffer(self, buffers: Optional[List[ReplayBuffer]]) -> None:
        """Check if the buffer matches the constraint."""
        if buffers is None:
            buffers = [VectorReplayBuffer(self.env_num, self.env_num) for _ in range(self.agt_num)]
        else:
            assert len(buffers) == self.agt_num
            for i in range(self.agt_num):
                if isinstance(buffers[i], ReplayBufferManager):
                    assert buffers[i].buffer_num >= self.env_num
                    if isinstance(buffers[i], CachedReplayBuffer):
                        assert buffers[i].cached_buffer_num >= self.env_num
                else:  # ReplayBuffer or PrioritizedReplayBuffer
                    assert buffers[i].maxsize > 0
                    if self.env_num > 1:
                        if type(buffers[i]) == ReplayBuffer:
                            buffer_type = "ReplayBuffer"
                            vector_type = "VectorReplayBuffer"
                        else:
                            buffer_type = "PrioritizedReplayBuffer"
                            vector_type = "PrioritizedVectorReplayBuffer"
                        raise TypeError(
                            f"Agent {i} cannot use {buffer_type}(size={buffers[i].maxsize}, ...) to collect "
                            f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                            f"{buffers[i].maxsize}, buffer_num={self.env_num}, ...) instead."
                        )
        self.buffers = buffers

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.datas = [Batch(obs={}, act={}, rew={}, done={},
                            obs_next={}, info={}, policy={}) for _ in range(self.agt_num)]
        self.reset_env()
        self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        for buffer in self.buffers:
            buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self) -> None:
        """Reset all of the environments."""
        obs = self.env.reset()
        # obs: (nenv, nagt) * dim -> nagt * (nenv, dim)
        self._set_obs(obs)

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        for data in self.datas:
            if hasattr(data.policy, "hidden_state"):
                state = data.policy.hidden_state  # it is a reference
                if isinstance(state, torch.Tensor):
                    state[id].zero_()
                elif isinstance(state, np.ndarray):
                    state[id] = None if state.dtype == object else 0
                elif isinstance(state, Batch):
                    state.empty_(id)
    
    def _set_obs(self, obs: np.ndarray) -> None:
        for i in range(self.agt_num):
            self.datas[i].obs = np.stack(obs[:, i])

    def _set_obs_next(self, obs_next: np.ndarray, idx: np.ndarray) -> None:
        for i in range(self.agt_num):
            self.datas[i].obs_next = np.stack(obs_next[idx, i])
    
    def _set_step_info(
        self, 
        obs_next: np.ndarray, 
        rew: np.ndarray, 
        done: np.ndarray,
        info: np.ndarray
    ) -> None:
        if len(rew.shape) == 1:
            # shared reward
            rew = np.tile(rew, (self.agt_num, 1)).T
        for i in range(self.agt_num):
            self.datas[i].obs_next = np.stack(obs_next[:, i])
            self.datas[i].rew = rew[:, i]
            self.datas[i].done = done
            self.datas[i].info = info
    
    def _update_obs(self) -> None:
        for i in range(self.agt_num):
            self.datas[i].obs = self.datas[i].obs_next

    def _set_mask(self, mask: np.ndarray) -> None:
        for i in range(self.agt_num):
            self.datas[i] = self.datas[i][mask]

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            for data in self.datas:
                data = data[:min(self.env_num, n_episode)]
        else:
            raise TypeError("Please specify at least one (either n_step or n_episode) "
                            "in AsyncCollector.collect().")

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        ep_rew = [None for _ in range(self.agt_num)]

        while True:
            actions = []
            # agents make decision sequentially
            for i in range(self.agt_num):
                assert len(self.datas[i]) == len(ready_env_ids)
                # restore the state: if the last state is None, it won't store
                last_state = self.datas[i].policy.pop("hidden_state", None)

                # get the next action
                if random:
                    self.datas[i].update(
                        act=[self._action_space[j][i].sample() for j in ready_env_ids])
                else:
                    if no_grad:
                        with torch.no_grad():
                            result = self.policies[i](self.datas[i], last_state)
                    else:
                        result = self.policies[i](self.datas[i], last_state)
                    # update state / act / policy into data
                    policy = result.get("policy", Batch())
                    assert isinstance(policy, Batch)
                    state = result.get("state", None)
                    if state is not None:
                        policy.hidden_state = state
                    act = to_numpy(result.act)
                    if self.exploration_noise:
                        act = self.policies[i].exploration_noise(act, self.datas[i])
                    self.datas[i].update(policy=policy, act=act)

                actions.append(self.policies[i].map_action(self.datas[i].act))
            
            actions = list(zip(*actions))
            # step in env
            obs_next, rew, done, info = self.env.step(actions, id=ready_env_ids)

            self._set_step_info(obs_next, rew, done, info)

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            for i in range(self.agt_num):
                ptr, ep_rew[i], ep_len, ep_idx = self.buffers[i].add(
                    self.datas[i], buffer_ids=ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                # episode_rews.append(ep_rew[env_ind_local])
                episode_rews.append(np.array(ep_rew)[:, env_ind_local].T)
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset = self.env.reset(env_ind_global)
                self._set_obs_next(obs_reset, env_ind_local)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self._set_mask(mask)
            self._update_obs()

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.datas = [Batch(obs={}, act={}, rew={}, done={},
                               obs_next={}, info={}, policy={}) for _ in range(self.agt_num)]
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(map(
                np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
        }