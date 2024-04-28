from __future__ import annotations

from typing import Dict, List

from fast_inference.algo.utils.env_info import EnvInfo
from fast_inference.cfg.configurable import Configurable
from fast_inference.utils.attr_dict import AttrDict


class VectorEnvRunner(Configurable):
    def __init__(self, cfg: AttrDict, env_info, worker_idx, split_idx):
        super().__init__(cfg)
        self.env_info: EnvInfo = env_info

        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step: int = 0  # current position in the rollout across all envs
        self.env_step_ready = False

    def process_policy_outputs(self, policy_inputs: List[Dict]):
        raise NotImplementedError()
    
    def advance_rollouts(self) -> List[Dict]:
        raise NotImplementedError()

    def generate_policy_inputs(self) -> List[Dict]:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()
