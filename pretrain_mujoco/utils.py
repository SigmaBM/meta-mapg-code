from numbers import Number
from typing import Any, AnyStr, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tianshou.utils.log_tools import BaseLogger
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

SIGMA_MIN = -20
SIGMA_MAX = 2


class RecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        name: str,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.device = device

        input_dim = int(np.prod(state_shape))
        output_dim = int(np.prod(action_shape))
        setattr(self, name + "_actor_l1", nn.Linear(input_dim, hidden_layer_size))
        setattr(self, name + "_actor_l2", nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True))
        setattr(self, name + "_actor_l3_mu", nn.Linear(hidden_layer_size, output_dim))
        
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            setattr(self, name + "_actor_l3_sigma", nn.Linear(hidden_layer_size, output_dim))
        else:
            setattr(self, name + "_actor_l3_sigma", nn.Parameter(torch.zeros(output_dim, 1)))
        
        self.name = name + "_actor"
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        # fc1 (before LSTM)
        s = getattr(self, self.name + "_l1")(s)
        # LSTM
        getattr(self, self.name + "_l2").flatten_parameters()
        if state is None:
            s, (h, c) = getattr(self, self.name + "_l2")(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = getattr(self, self.name + "_l2")(s, (state["h"].transpose(0, 1).contiguous(),
                                                             state["c"].transpose(0, 1).contiguous()))
        logits = s[:, -1]
        # fc2 - mu
        mu = getattr(self, self.name + "_l3_mu")(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                getattr(self, self.name + "_l3_sigma")(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (getattr(self, self.name + "_l3_sigma").view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {"h": h.transpose(0, 1).detach(),
                             "c": c.transpose(0, 1).detach()}


class RecurrentCritic(nn.Module):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        name: str,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = [0],
        hidden_layer_size: int = 128,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        
        setattr(self, name + "_value_l1", nn.Linear(int(np.prod(state_shape)), hidden_layer_size))
        setattr(self, name + "_value_l2", nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True))
        setattr(self, name + "_value_l3", nn.Linear(hidden_layer_size + int(np.prod(action_shape)), 1))

        self.name = name + "_value"

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        # fc1 (before LSTM)
        s = getattr(self, self.name + "_l1")(s)
        # LSTM
        getattr(self, self.name + "_l2").flatten_parameters()
        s, (h, c) = getattr(self, self.name + "_l2")(s)
        s = s[:, -1]
        if a is not None:
            a = torch.as_tensor(
                a, device=self.device, dtype=torch.float32)  # type: ignore
            s = torch.cat([s, a], dim=1)
        value = getattr(self, self.name + "_l3")(s)

        return value


class BasicLogger(BaseLogger):
    """A loggger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    You can also rewrite write() func to use your own writer.

    :param SummaryWriter writer: the writer to log data.
    :param int train_interval: the log interval in log_train_data(). Default to 1.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1,
        test_interval: int = 1,
        update_interval: int = 1000,
    ) -> None:
        super().__init__(writer)
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1

    def write(
        self, key: str, x: int, y: Union[Number, np.number, np.ndarray], **kwargs: Any
    ) -> None:
        self.writer.add_scalar(key, y, global_step=x)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew" and "len" keys.
        """
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean(axis=0)
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:
                self.write("train/n/ep", step, collect_result["n/ep"])
                self.write("train/len", step, collect_result["len"])
                for i in range(len(collect_result["rew"])):
                    self.write("train/rew_{}".format(i), step, collect_result["rew"][i])
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew", "rew_std", "len",
            and "len_std" keys.
        """
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(axis=0), rews.std(axis=0), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            self.write("test/len", step, len_)
            self.write("test/len_std", step, len_std)
            for i in range(len(collect_result["rew"])):
                self.write("test/rew_{}".format(i), step, rew[i])
                self.write("test/rew_{}_std".format(i), step, rew_std[i]) 
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            for k, v in update_result.items():
                self.write(k, step, v)
            self.last_log_update_step = step
