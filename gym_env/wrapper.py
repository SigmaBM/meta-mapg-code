from argparse import Namespace
import os.path as osp
from typing import Any

import gym, os
import numpy as np


class Wrapper(object):
    """A environment wrapper to add sample persona function.
    
        env: gym environment to be wrappered.
    """
    def __init__(self, env: gym.Env, args: Namespace) -> None:
        super().__init__()
        self.env = env
        if args.model_path is None:
            self.path = osp.join(".", "pretrain_model", env.scenario)
        else:
            self.path = args.model_path
        self.list_tr = args.list_tr
        self.list_va = args.list_va
        self.list_ts = args.list_ts
    
    def __getattr__(self, key: str) -> Any:
        return getattr(self.env, key)
    
    @property
    def unwrapped(self) -> gym.Env:
        return self.env
    
    def sample_personas(self, is_train, is_val=True, **kwargs):
        if is_train:
            persona = {}
            persona["iteration"] = np.random.choice(self.list_tr, 1)[0]
            persona["filepath"] = osp.join(self.path, persona["iteration"])
            return [persona]
        if is_val:
            personas = [{
                "iteration": it,
                "filepath": osp.join(self.path, it)
            } for it in self.list_va]
        else:
            personas = [{
                "iteration": it,
                "filepath": osp.join(self.path, it)
            } for it in self.list_ts]
        return personas
