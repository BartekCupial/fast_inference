from __future__ import annotations

import random
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from fast_inference.algo.sampling.sampling_utils import VectorEnvRunner
from fast_inference.algo.utils.make_env import make_env_func_non_batched
from fast_inference.algo.utils.tensor_utils import ensure_numpy_array
from fast_inference.utils.attr_dict import AttrDict
from fast_inference.algo.utils.env_info import EnvInfo


class ActorState:
    """
    State of a single actor (agent) in a multi-agent environment.
    Single-agent environments are treated as multi-agent with one agent for simplicity.
    """
    
    def __init__(
        self,
        env_info: EnvInfo,
        worker_idx: int,
        split_idx: int,
        env_idx: int,
        agent_idx: int,
        global_env_idx: int,
    ):
        self.env_info = env_info
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.env_idx = env_idx
        self.agent_idx = agent_idx
        self.global_env_idx: int = global_env_idx  # global index of the policy in the entire system

        self.last_actions = None
        self.last_obs = None
        
        self.num_trajectories = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0

    def curr_actions(self) -> np.ndarray | List | Any:
        """
        :return: the latest set of actions for this actor, calculated by the policy worker for the last observation
        """
        actions = ensure_numpy_array(self.last_actions)

        if self.env_info.all_discrete or isinstance(self.env_info.action_space, gym.spaces.Discrete):
            return self._process_action_space(actions, is_discrete=True)
        elif isinstance(self.env_info.action_space, gym.spaces.Box):
            return self._process_action_space(actions, is_discrete=False)
        elif isinstance(self.env_info.action_space, gym.spaces.Tuple):
            out_actions = []
            for split, space in zip(
                np.split(actions, np.cumsum(self.env_info.action_splits)[:-1]), self.env_info.action_space
            ):
                is_discrete = isinstance(space, gym.spaces.Discrete)
                out_actions.append(self._process_action_space(split, is_discrete))
            return out_actions

        raise NotImplementedError(f"Unknown action space type: {type(self.env_info.action_space)}")

    def _episodic_stats(self, info: Dict) -> Dict[str, Any]:
        stats = dict(
            reward=self.last_episode_reward,
            len=self.last_episode_duration,
            episode_extra_stats=info.get("episode_extra_stats", dict()),
        )

        if (true_objective := info.get("true_objective", self.last_episode_reward)) is not None:
            stats["true_objective"] = true_objective

        return stats

    def record_env_step(self, reward, terminated: bool, truncated: bool, info, rollout_step):
        """
        Policy inputs (obs) and policy outputs (actions, values, ...) for the current rollout step
        are already added to the trajectory buffer
        the only job remaining is to add auxiliary data: rewards, done flags, etc.

        :param reward: last reward from the env step
        :param terminated: whether the episode has terminated
        :param truncated: whether the episode has been truncated (i.e. max episode length reached)
        :param info: info dictionary
        :param rollout_step: number of steps since we started the current rollout. When this reaches cfg.rollout
        we finalize the trajectory buffer and send it to the learner.
        """

        done = terminated | truncated

        report = None
        if done:
            report = self._episodic_stats(info)

            self.last_episode_reward = self.last_episode_duration = 0.0

        return report


class NonBatchedVectorEnvRunner(VectorEnvRunner):
    def __init__(
        self,
        cfg,
        env_info,
        num_envs,
        worker_idx,
        split_idx,
    ):
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        :param training_info: curr env steps, reward shaping scheme, etc.
        """
        super().__init__(cfg, env_info, worker_idx, split_idx)

        self.num_envs = num_envs
        self.num_agents = 1

        self.envs, self.episode_rewards = [], []
        self.actor_states: List[List[ActorState]] = []

        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            global_env_idx = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx,
                vector_index=vector_idx,
                env_id=global_env_idx,
            )

            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            env = make_env_func_non_batched(self.cfg, env_config=env_config)

            self.envs.append(env)
            
            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(
                    self.env_info,
                    self.worker_idx,
                    self.split_idx,
                    env_i,
                    agent_idx,
                    global_env_idx,
                )
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

        self._reset()

    def _reset(self):
        """
        Do the very first reset for all environments in a vector. Populate shared memory with initial obs.
        Note that this is called only once, at the very beginning of training. After this the envs should auto-reset.

        :return: first requests for policy workers (to generate actions for the very first env step)
        """

        for env_i, e in enumerate(self.envs):
            seed = env_i
            observations, info = e.reset(seed=seed)  # new way of doing seeding since Gym 0.26.0

            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.last_obs = obs

        self.env_step_ready = True

    def process_policy_outputs(self, policy_outputs):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                # TODO: save here policy outputs
                actor_state.last_actions = policy_outputs_dict["actions"].squeeze()

    def _process_rewards(self, rewards, env_i: int):
        """
        Pretty self-explanatory, here we record the episode reward and apply the optional clipping and
        scaling of rewards.
        """
        for agent_i, r in enumerate(rewards):
            self.actor_states[env_i][agent_i].last_episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = rewards * self.cfg.reward_scale
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        return rewards

    def _process_env_step(self, new_obs, rewards, terminated, truncated, infos, env_i):
        episodic_stats = []
        env_actor_states = self.actor_states[env_i]

        rewards = self._process_rewards(rewards, env_i)

        for agent_i in range(self.num_agents):
            actor_state = env_actor_states[agent_i]

            episode_report = actor_state.record_env_step(
                rewards[agent_i],
                terminated[agent_i],
                truncated[agent_i],
                infos[agent_i],
                self.rollout_step,
            )

            actor_state.last_obs = new_obs[agent_i]

            if episode_report:
                episodic_stats.append(episode_report)

        return episodic_stats

    def _prepare_next_step(self):
        policy_inputs = []
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                policy_inputs.append(dict(obs=actor_state.last_obs))
                
        assert False
        # TODO: batching, tokenization, etc
        return policy_inputs

    def advance_rollouts(self) -> Tuple[List[Dict], List[Dict]]:
        episodic_stats = []

        for env_i, e in enumerate(self.envs):
            actions = [s.curr_actions() for s in self.actor_states[env_i]]
            new_obs, rewards, terminated, truncated, infos = e.step(actions)

            stats = self._process_env_step(new_obs, rewards, terminated, truncated, infos, env_i)
            episodic_stats.extend(stats)

        self.rollout_step += 1

        return episodic_stats

    def generate_policy_inputs(self) -> List[Dict]:
        policy_inputs = self._prepare_next_step()
        return policy_inputs

    def close(self):
        for e in self.envs:
            e.close()
