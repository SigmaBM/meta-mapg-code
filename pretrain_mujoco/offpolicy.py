import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import tqdm
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config

from pretrain_mujoco.macollector import MACollector


def offpolicy_trainer(
    policies: List[BasePolicy],
    train_collector: MACollector,
    test_collector: MACollector,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
    test_in_train: bool = True,
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policies: a list of the :class:`~tianshou.policy.BasePolicy` class.
    :param MACollector train_collector: the collector used for training.
    :param MACollector test_collector: the collector used for testing.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatly in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int/float update_per_step: the number of times the policy network would be
        updated per transition after (step_per_collect) transitions are collected,
        e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will
        be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
        collected by the collector. Default to 1.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.

    References:
        https://github.com/thu-ml/tianshou/tianshou/trainer/offpolicy.py
    """
    env_step, gradient_step = 0, 0
    last_rew = [0. for _ in range(len(policies))]
    last_len = 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_result = test_episode(policies, test_collector, 0, episode_per_test,
                               logger, env_step)
    for epoch in range(1, 1 + max_epoch):
        # train
        [pi.train() for pi in policies]
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                result = train_collector.collect(n_step=step_per_collect)
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_len = result['len'] if 'len' in result else last_len
                data = {
                    "env_step": str(env_step),
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                for i in range(len(policies)):
                    last_rew[i] = result['rew'][i] if 'rew' in result else last_rew[i]
                    data["rew_{}".format(i)] = f"{last_rew[i]:.2f}"
                for i in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses_all = {}
                    for j in range(len(policies)):
                        losses = policies[j].update(batch_size, train_collector.buffers[j])
                        for k in losses.keys():
                            k_id = "agent_{}/{}".format(j, k)
                            stat[k_id].add(losses[k])
                            losses_all[k_id] = stat[k_id].get()
                            data[k_id] = f"{losses[k]:.3f}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)

            if t.n <= t.total:
                t.update()
        # test
        test_result = test_episode(policies, test_collector, epoch,
                                   episode_per_test, logger, env_step)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if verbose:
            print(f"Epoch #{epoch}:")
            for i in range(len(policies)):
                print(f"Agent {i} - test_reward: {rew[i]:.6f} ± {rew_std[i]:.6f};")

    return gather_info(start_time, train_collector, test_collector)


def test_episode(
    policies: List[BasePolicy],
    collector: MACollector,
    epoch: int,
    n_episode: int,
    logger: Optional[BaseLogger] = None,
    global_step: Optional[int] = None,
) -> Dict[str, Any]:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    [pi.eval() for pi in policies]
    result = collector.collect(n_episode=n_episode)
    if logger and global_step is not None:
        logger.log_test_data(result, global_step)
    return result


def gather_info(
    start_time: float,
    train_c: Optional[MACollector],
    test_c: MACollector,
) -> Dict[str, Union[float, str]]:
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting transitions in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (env_step per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (env_step per second);
        * ``duration`` the total elapsed time.
    """
    duration = time.time() - start_time
    model_time = duration - test_c.collect_time
    test_speed = test_c.collect_step / test_c.collect_time
    result: Dict[str, Union[float, str]] = {
        "test_step": test_c.collect_step,
        "test_episode": test_c.collect_episode,
        "test_time": f"{test_c.collect_time:.2f}s",
        "test_speed": f"{test_speed:.2f} step/s",
        "duration": f"{duration:.2f}s",
        "train_time/model": f"{model_time:.2f}s",
    }
    if train_c is not None:
        model_time -= train_c.collect_time
        train_speed = train_c.collect_step / (duration - test_c.collect_time)
        result.update({
            "train_step": train_c.collect_step,
            "train_episode": train_c.collect_episode,
            "train_time/collector": f"{train_c.collect_time:.2f}s",
            "train_time/model": f"{model_time:.2f}s",
            "train_speed": f"{train_speed:.2f} step/s",
        })
    return result