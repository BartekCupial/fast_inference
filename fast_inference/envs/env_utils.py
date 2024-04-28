from __future__ import annotations

from functools import wraps
from time import sleep
from typing import Any, Dict, Optional

from fast_inference.algo.utils.context import global_env_registry
from fast_inference.utils.typing import CreateEnvFunc
from fast_inference.utils.utils import is_module_available, log


def register_env(env_name: str, make_env_func: CreateEnvFunc) -> None:
    """
    Register a callable that creates an environment.
    This callable is called like:
        make_env_func(full_env_name, cfg, env_config)
        Where full_env_name is the name of the environment to be created, cfg is a namespace or AttrDict containing
        necessary configuration parameters and env_config is an auxiliary dictionary containing information such as worker index on which the environment lives
        (some envs may require this information)
    env_name: name of the environment
    make_env_func: callable that creates an environment
    """

    env_registry = global_env_registry()

    if env_name in env_registry:
        log.warning(f"Environment {env_name} already registered, overwriting...")

    assert callable(make_env_func), f"{make_env_func=} must be callable"

    env_registry[env_name] = make_env_func


def find_wrapper_interface(env, interface_type):
    """Unwrap the env until we find the wrapper that implements interface_type."""
    unwrapped = env.unwrapped
    while True:
        if isinstance(env, interface_type):
            return env
        elif env == unwrapped:
            return None  # unwrapped all the way and didn't find the interface
        else:
            env = env.env  # unwrap by one layer


def num_env_steps(infos):
    """Calculate number of environment frames in a batch of experience."""

    total_num_frames = 0
    for info in infos:
        total_num_frames += info.get("num_frames", 1)
    return total_num_frames
