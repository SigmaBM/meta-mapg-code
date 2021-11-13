import os.path as osp
from typing import Any

import gym, os
import numpy as np


class Wrapper(object):
    """A environment wrapper to add sample persona function.
    
        env: gym environment to be wrappered.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.env = env
        self.path = osp.join(".", "pretrain_model", env.scenario)

        # only support agent 1 in 2-agent env for now
        self._get_personas()

    def _get_personas(self) -> None:
        self.tr_paths = os.listdir(osp.join(self.path, "train"))
        self.va_paths = os.listdir(osp.join(self.path, "valid"))
        self.ts_paths = os.listdir(osp.join(self.path, "test"))
    
    def __getattr__(self, key: str) -> Any:
        return getattr(self.env, key)
    
    @property
    def unwrapped(self) -> gym.Env:
        return self.env
    
    def sample_personas(self, is_train, is_val=True, **kwargs):
        if is_train:
            persona = {}
            persona["iteration"] = np.random.choice(self.tr_paths, 1)[0]
            persona["filepath"] = osp.join(self.path, "train", persona["iteration"])
            return [persona]
        if is_val:
            personas = [{
                "iteration": it,
                "filepath": osp.join(self.path, "valid", it)
            } for it in self.va_paths]
        else:
            personas = [{
                "iteration": it,
                "filepath": osp.join(self.path, "test", it)
            } for it in self.ts_paths]
        return personas
