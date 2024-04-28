from __future__ import annotations

import multiprocessing
import os
import pickle
from dataclasses import dataclass
from os.path import join

import gymnasium as gym

from fast_inference.algo.utils.context import set_global_context, sf_global_context
from fast_inference.algo.utils.make_env import NonBatchedVecEnv, make_env_func_non_batched
from fast_inference.utils.typing import Config
from fast_inference.utils.utils import log, project_tmp_dir

ENV_INFO_PROTOCOL_VERSION = 1


@dataclass
class EnvInfo:
    obs_space: gym.Space
    action_space: gym.Space
    num_agents: int


def extract_env_info(env: NonBatchedVecEnv) -> EnvInfo:
    obs_space = env.observation_space
    action_space = env.action_space
    num_agents = env.num_agents

    env_info = EnvInfo(
        obs_space=obs_space,
        action_space=action_space,
        num_agents=num_agents,
    )
    return env_info


def spawn_tmp_env_and_get_info(sf_context, res_queue, cfg):
    set_global_context(sf_context)

    tmp_env = make_env_func_non_batched(cfg, env_config=None)
    env_info = extract_env_info(tmp_env)
    tmp_env.close()
    del tmp_env

    log.debug("Env info: %r", env_info)
    res_queue.put(env_info)


def env_info_cache_filename(cfg: Config) -> str:
    return join(project_tmp_dir(), f"env_info_{cfg.env}")


def obtain_env_info_in_a_separate_process(cfg: Config) -> EnvInfo:
    cache_filename = env_info_cache_filename(cfg)
    if cfg.use_env_info_cache and os.path.isfile(cache_filename):
        log.debug(f"Loading env info from cache: {cache_filename}")
        with open(cache_filename, "rb") as fobj:
            env_info = pickle.load(fobj)
            if env_info.env_info_protocol_version == ENV_INFO_PROTOCOL_VERSION:
                return env_info

    sf_context = sf_global_context()

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=spawn_tmp_env_and_get_info, args=(sf_context, q, cfg))
    p.start()

    env_info = q.get()
    p.join()

    if cfg.use_env_info_cache:
        with open(cache_filename, "wb") as fobj:
            pickle.dump(env_info, fobj)

    return env_info
