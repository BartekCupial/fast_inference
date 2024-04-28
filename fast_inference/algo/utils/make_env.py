from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import Wrapper, spaces
from gymnasium.core import ActType, ObsType
from torch import Tensor

from fast_inference.algo.utils.tensor_utils import dict_of_lists_cat
from fast_inference.envs.create_env import create_env
from fast_inference.utils.dicts import dict_of_lists_append, list_of_dicts_to_dict_of_lists
from fast_inference.utils.typing import Config

Actions = Any
ListActions = Sequence[Actions]
TensorActions = Tensor

SeqBools = Sequence[bool]

DictObservations = Dict[str, Any]
DictOfListsObservations = Dict[str, Sequence[Any]]
DictOfTensorObservations = Dict[str, Tensor]
ListObservations = Sequence[Any]
ListOfDictObservations = Sequence[DictObservations]


def get_multiagent_info(env: Any) -> Tuple[bool, int]:
    num_agents = env.num_agents if hasattr(env, "num_agents") else 1
    is_multiagent = env.is_multiagent if hasattr(env, "is_multiagent") else num_agents > 1
    assert is_multiagent or num_agents == 1, f"Invalid configuration: {is_multiagent=} and {num_agents=}"
    return is_multiagent, num_agents


def is_multiagent_env(env: Any) -> bool:
    is_multiagent, num_agents = get_multiagent_info(env)
    return is_multiagent


class _DictObservationsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        is_multiagent, num_agents = get_multiagent_info(env)
        self.is_multiagent: bool = is_multiagent
        self.num_agents: int = num_agents
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(dict(obs=self.observation_space))


class NonBatchedMultiAgentWrapper(Wrapper):
    """
    This wrapper allows us to treat a single-agent environment as multi-agent with 1 agent.
    That is, the data (obs, rewards, etc.) is converted into lists of length 1
    """

    def __init__(self, env):
        super().__init__(env)

        self.num_agents: int = 1
        self.is_multiagent: bool = True

    def reset(self, **kwargs) -> ListObservations:
        obs, info = self.env.reset(**kwargs)
        return [obs], [info]

    def step(self, action: ListActions) -> Tuple[ListObservations, Sequence, SeqBools, SeqBools, Sequence[Dict]]:
        action = action[0]
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:  # auto-resetting
            obs, info["reset_info"] = self.env.reset()
        return [obs], [rew], [terminated], [truncated], [info]


class NonBatchedDictObservationsWrapper(_DictObservationsWrapper):
    """Guarantees that the environment returns observations as lists of dictionaries."""

    def reset(self, **kwargs) -> ListOfDictObservations:
        obs, info = self.env.reset(**kwargs)
        return [dict(obs=o) for o in obs], info

    def step(self, action: ListActions) -> Tuple[ListOfDictObservations, Any, Any, Any, Any]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        return [dict(obs=o) for o in obs], rew, terminated, truncated, info


class NonBatchedVecEnv(Wrapper):
    """
    reset() returns a list of dict observations.
    step(action) returns a list of dict observations, list of rewards, list of dones, list of infos.
    """

    def __init__(self, env):
        if not is_multiagent_env(env):
            env = NonBatchedMultiAgentWrapper(env)
        if not isinstance(env.observation_space, spaces.Dict):
            env = NonBatchedDictObservationsWrapper(env)

        is_multiagent, num_agents = get_multiagent_info(env)
        self.is_multiagent: bool = is_multiagent
        self.num_agents: int = num_agents
        super().__init__(env)


def make_env_func_non_batched(cfg: Config, env_config, render_mode: Optional[str] = None) -> NonBatchedVecEnv:
    """
    This should yield an environment that always returns a list of {observations, rewards,
    dones, etc.}
    This is for the non-batched sampler which processes each agent's data independently without any vectorization
    (and therefore enables more sophisticated configurations where agents in the same env can be controlled
    by different policies and so on).
    """
    env = create_env(cfg.env, cfg=cfg, env_config=env_config, render_mode=render_mode)
    env = NonBatchedVecEnv(env)
    return env
