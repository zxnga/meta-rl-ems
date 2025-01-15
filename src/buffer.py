from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union, List, Callable, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

class HistoryMultipleReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    history: th.Tensor

class MultipleReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_task: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_task, optimize_memory_usage, handle_timeout_termination)
        self.n_task = n_task
        self.buffer_size = buffer_size
        self.pos = np.zeros((n_task), dtype=int)
        self.full = np.zeros((n_task))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        task_id: int, #position of task in the buffer
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            raise NotImplementedError

        # # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        # action = action.reshape((self.n_task, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos[task_id]][task_id] = np.array(obs)

        if self.optimize_memory_usage:
            raise NotImplementedError
            # self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos[task_id]][task_id] = np.array(next_obs)

        self.actions[self.pos[task_id]][task_id] = np.array(action)
        self.rewards[self.pos[task_id]][task_id] = np.array(reward)
        self.dones[self.pos[task_id]][task_id] = np.array(done)

        if self.handle_timeout_termination:
            raise NotImplementedError
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos[task_id] += 1
        if self.pos[task_id] == self.buffer_size:
            self.full[task_id] = True
            self.pos[task_id] = 0

    def sample(self, batch_size: int, task_id: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full[task_id] else self.pos[task_id]
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, task_id, env=env)

    def _get_samples(self, batch_inds: np.ndarray, task_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample only from task_id buffer
        env_indices = np.ones(10, dtype=int) * task_id

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class HistoryMultipleReplayBuffer(MultipleReplayBuffer):
    def __init__(
        self,
        history_window: int,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_task: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_task, optimize_memory_usage, handle_timeout_termination)
        self.history_window = history_window

    def get_valid_history_start(self, batch_inds, task_id):
        """
        1. remove history indices that are outside of buffer's range (<0)
        2. remove indices that are before the last reset of the environment (based on the dones buffer)
        """
        updated_history_indices = []
        #1
        history_indices = [range(max(0,i-self.history_window),i) for i in batch_inds]
        #2
        for h in history_indices:
            idx = np.where(self.dones[h.start:h.stop + 1, task_id] == 1)[0]
            if idx.size > 0:
                idx = np.where(self.dones[h.start:h.stop + 1, task_id] == 1)[0].max()
                idx += h.start
                updated_history_indices.append(range(max(h.start, idx),h.stop))
            else:
                updated_history_indices.append(h)
        return updated_history_indices

    def pad_and_slice(self, array, task_id, idx_range, pad_value=0):
        """
        Pads and slices the array based on history indices, ensuring consistency in shape.
        """
        n_pads = self.history_window - len(idx_range)
        pad_width = [(n_pads, 0)] + [(0, 0)] * (array.ndim - 1)
        padded_array = np.pad(array, pad_width, constant_values=pad_value)
        return padded_array[idx_range.start:idx_range.stop+n_pads, task_id, :]

    def collect_history(self, batch_inds: np.ndarray, task_id: int, env: Optional[VecNormalize] = None):
        obs_history = []
        action_history = []
        reward_history = []

        history_indices = self.get_valid_history_start(batch_inds, task_id)
        for idx_range in history_indices:
            obs_history.append(self.pad_and_slice(self.observations, task_id, idx_range))
            action_history.append(self.pad_and_slice(self.actions, task_id, idx_range))
            reward_history.append(self.pad_and_slice(np.expand_dims(self.rewards, axis=2), task_id, idx_range))
        obs_history = np.array(obs_history)
        action_history = np.array(action_history)
        reward_history = np.array(reward_history)
        # reward_history = np.expand_dims(np.array(reward_history), axis=2) #(batch_size,history_window) to (batch_size,history_window,1)

        return np.concatenate((obs_history, action_history, reward_history), axis=2) #(batch_size, history_window, obs_dim+action_dim+reward_dim)

    def _get_samples(self, batch_inds: np.ndarray, task_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample only from task_id buffer
        print(batch_inds)
        env_indices = np.ones(batch_inds.size, dtype=int) * task_id

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.collect_history(batch_inds, task_id, env),
        )
        return HistoryMultipleReplayBufferSamples(*tuple(map(self.to_torch, data)))