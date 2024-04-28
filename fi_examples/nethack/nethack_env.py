from pathlib import Path
from typing import Optional

from nle.env.tasks import NetHackScore, NetHackChallenge, NetHackEat, NetHackGold, NetHackOracle, NetHackScout, NetHackStaircase, NetHackStaircasePet

from fi_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    NLETimeLimit,
    NLELMWrapper,
    TaskRewardsInfoWrapper,
    TtyrecInfoWrapper,
)

NETHACK_ENVS = dict(
    nethack_staircase=NetHackStaircase,
    nethack_score=NetHackScore,
    nethack_pet=NetHackStaircasePet,
    nethack_oracle=NetHackOracle,
    nethack_gold=NetHackGold,
    nethack_eat=NetHackEat,
    nethack_scout=NetHackScout,
    nethack_challenge=NetHackChallenge,
)


def nethack_env_by_name(name):
    if name in NETHACK_ENVS.keys():
        return NETHACK_ENVS[name]
    else:
        raise Exception("Unknown NetHack env")


def make_nethack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    env_class = nethack_env_by_name(env_name)

    observation_keys = (
        "blstats",
        "tty_chars",
        "tty_cursor",
        "glyphs",
        "inv_strs",
        "inv_letters",
    )

    if cfg.gameloaddir:
        # gameloaddir can be either list or a single path
        if isinstance(cfg.gameloaddir, list):
            # if gameloaddir is a list we will pick only one element from this list
            if env_config:
                # based on env_id
                idx = env_config["env_id"] % len(cfg.gameloaddir)
                gameloaddir = cfg.gameloaddir[idx]
            else:
                # if no env_id use first element
                gameloaddir = cfg.gameloaddir[0]
        else:
            # if gameliaddir is a single path
            assert isinstance(cfg.gameloaddir, (str, Path))
            gameloaddir = cfg.gameloaddir
    else:
        gameloaddir = None

    kwargs = dict(
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
        gameloaddir=gameloaddir,
    )
    if env_name == "challenge":
        kwargs["no_progress_timeout"] = cfg.no_progress_timeout

    if env_name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")
    if cfg.state_counter is not None:
        kwargs.update(state_counter=cfg.state_counter)

    env = env_class(**kwargs)
    env = NLELMWrapper(env, observation=True, random_template=False, include_interleave_in_prompt=False)

    # add TimeLimit.truncated to info
    env = NLETimeLimit(env)

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env)
        if cfg.save_ttyrec_every != 0:
            env = TtyrecInfoWrapper(env)

    return env
