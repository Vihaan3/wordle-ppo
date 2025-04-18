import os
import sys
import time
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal, Union, Optional

import numpy as np
from numpy.random import Generator
import torch as t
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym
from gymnasium import spaces
import gymnasium.envs.registration
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from tqdm import tqdm
import einops
import wandb
import string
import collections
import matplotlib.pyplot as plt

from IPython.display import clear_output
from matplotlib.animation import FuncAnimation
from jaxtyping import Float, Int


warnings.filterwarnings('ignore')
Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def load_words(n: int = None, path: str = "wordle_words.txt", seed: int = None) -> list[str]:
    """
    Load a word list from file and return a random sample of `n` words.
    If `n` is None, returns the full list.
    """
    with open(path, "r") as f:
        lines = [line.strip().upper() for line in f.readlines()]
        words = [word for word in lines if len(word) == 5 and word.isalpha()]

        rng = random.Random(seed) if seed is not None else random
        if n is not None and n < len(words):
            return rng.sample(words, n)
        return words

"""Env Code: Heavily inspired by Andrew Kho: 
https://github.com/andrewkho/wordle-solver/tree/4495ae13ca31ae0f9784b847e34d7ef4117a1819/deep_rl/wordle"""

WORDLE_CHARS = string.ascii_uppercase  
WORDLE_N = 5  
REWARD_WIN = 20
REWARD_REPEAT_PENALTY = -30

CharIndex = {c: i for i, c in enumerate(WORDLE_CHARS)}

def get_state_shape(max_turns: int):
    return [max_turns] + [2] * len(WORDLE_CHARS) + [2] * 3 * WORDLE_N * len(WORDLE_CHARS)

def new_state(max_turns: int) -> np.ndarray:
    return np.array(
        [max_turns] + [0] * len(WORDLE_CHARS) + [0, 1, 0] * WORDLE_N * len(WORDLE_CHARS),
        dtype=np.int32
    )

def remaining_steps(state: np.ndarray) -> int:
    return state[0]

NO = 0
SOMEWHERE = 1
YES = 2

def get_mask(word: str, goal_word: str) -> List[int]:
    mask = [0] * len(word)
    goal_char_counts = collections.Counter(goal_word)

    for i in range(len(word)):
        if word[i] == goal_word[i]:
            mask[i] = 2
            goal_char_counts[word[i]] -= 1

    for i in range(len(word)):
        if mask[i] != 0:
            continue
        if word[i] in goal_char_counts and goal_char_counts[word[i]] > 0:
            mask[i] = 1
            goal_char_counts[word[i]] -= 1

    return mask

def update_from_mask(state: np.ndarray, word: str, mask: List[int]) -> np.ndarray:
    state = state.copy()
    state[0] -= 1

    prior_yes = []
    prior_maybe = []

    for i, c in enumerate(word):
        cint = CharIndex[c]
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        state[1 + cint] = 1
        if mask[i] == YES:
            prior_yes.append(c)
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(WORDLE_CHARS)):
                if ocint != cint:
                    oc_offset = 1 + len(WORDLE_CHARS) + ocint * WORDLE_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]

    for i, c in enumerate(word):
        cint = CharIndex[c]
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        if mask[i] == SOMEWHERE:
            prior_maybe.append(c)
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        elif mask[i] == NO:
            if c in prior_maybe:
                state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
            elif c in prior_yes:
                for j in range(WORDLE_N):
                    if state[offset + 3 * j + 1] == 1:
                        state[offset + 3 * j:offset + 3 * j + 3] = [1, 0, 0]
            else:
                state[offset:offset + 3 * WORDLE_N] = [1, 0, 0] * WORDLE_N

    return state

class MinimalWordleEnv(gym.Env):
    def __init__(self, words: List[str], max_turns: int = 6):
        super().__init__()
        self.words = words
        self.max_turns = max_turns
        self.action_space = spaces.Discrete(len(self.words))

        self.observation_space = spaces.Dict({
            "state": spaces.MultiDiscrete(get_state_shape(max_turns)),
            "guessed": spaces.MultiBinary(len(self.words)),
        })

        self.state = None
        self.guessed = None
        self.goal_word = None
        self.done = True
        self.remaining_candidates = set(words)
        self.prev_entropy = None
        self.history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.state = new_state(self.max_turns)
        self.guessed = np.zeros(len(self.words), dtype=np.int32)
        self.done = False
        self.goal_word = self.np_random.choice(self.words)
        self.remaining_candidates = set(self.words)
        self.prev_entropy = np.log(len(self.remaining_candidates))
        self.history = []
        return {"state": self.state.copy(), "guessed": self.guessed.copy()}, {}

    def step(self, action: int):
        guessed_word = self.words[action]
        self.guessed[action] = 1
        mask = get_mask(guessed_word, self.goal_word)
        self.state = update_from_mask(self.state, guessed_word, mask)

        self.remaining_candidates = set([
            w for w in self.remaining_candidates if get_mask(guessed_word, w) == mask
        ])
        new_entropy = np.log(len(self.remaining_candidates)) if len(self.remaining_candidates) > 0 else 0.0

        entropy_delta = self.prev_entropy - new_entropy
        self.prev_entropy = new_entropy
        reward = entropy_delta * 10

        if entropy_delta == 0 or guessed_word in [g for g, _ in self.history]:
            reward += REWARD_REPEAT_PENALTY  

        green_bonus = sum([m == 2 for m in mask])
        reward += green_bonus

        terminated = guessed_word == self.goal_word
        truncated = remaining_steps(self.state) == 0 and not terminated

        if terminated:
            reward += REWARD_WIN

        self.done = terminated or truncated
        self.history.append((guessed_word, mask))

        return {"state": self.state.copy(), "guessed": self.guessed.copy()}, reward, terminated, truncated, {
            "goal_word": self.goal_word,
            "guess": guessed_word,
            "mask": mask,
            "remaining_candidates": len(self.remaining_candidates),
            "entropy": self.prev_entropy
        }

class WordleEnv10(MinimalWordleEnv):
    def __init__(self):
        super().__init__(words=load_words(10), max_turns=6)

class WordleEnv100(MinimalWordleEnv):
    def __init__(self):
        super().__init__(words=load_words(100), max_turns=6)

class WordleEnv500(MinimalWordleEnv):
    def __init__(self):
        super().__init__(words=load_words(500), max_turns=6)

class WordleEnv1000(MinimalWordleEnv):
    def __init__(self):
        super().__init__(words=load_words(1000), max_turns=6)

class WordleEnvFull(MinimalWordleEnv):
    def __init__(self):
        super().__init__(words=load_words(None), max_turns=6)  
		
def register_all_wordle_envs():
    gym.envs.registration.register(
        id="Wordle10-v0", entry_point=WordleEnv10
    )
    gym.envs.registration.register(
        id="Wordle100-v0", entry_point=WordleEnv100
    )
    gym.envs.registration.register(
        id="Wordle500-v0", entry_point=WordleEnv500
    )
    gym.envs.registration.register(
        id="Wordle1000-v0", entry_point=WordleEnv1000
    )
    gym.envs.registration.register(
        id="WordleFull-v0", entry_point=WordleEnvFull
    )
	
def preprocess_obs(obs: dict | t.Tensor | np.ndarray) -> t.Tensor:
    """
    Preprocess observation:
    - Converts to float32, moves to device.
    - Normalizes feature ranges
    """
    if isinstance(obs, dict):
        processed = []

        if "state" in obs:
            state = obs["state"]
            if isinstance(state, np.ndarray):
                state = t.tensor(state, dtype=t.float32, device=device)
            else:
                state = state.to(dtype=t.float32, device=device)
            state = state / 2.0
            processed.append(state)

        if "guessed" in obs:
            guessed = obs["guessed"]
            if isinstance(guessed, np.ndarray):
                guessed = t.tensor(guessed, dtype=t.float32, device=device)
            else:
                guessed = guessed.to(dtype=t.float32, device=device)
            processed.append(guessed)

        return t.cat(processed, dim=-1)

    if isinstance(obs, np.ndarray):
        obs = t.tensor(obs, dtype=t.float32, device=device)
    else:
        obs = obs.to(dtype=t.float32, device=device)

    return obs / 2.0
	
def get_inner_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

"""PPO Implementation code, heavily inspired by ARENA's Implementation but tweaked for Wordle:
https://arena-chapter2-rl.streamlit.app/. Original comments have been mostly left in and additional comments have been added
to describe the most important tweaks."""
@dataclass
class PPOArgs:

    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    env_id: str = "Wordle"
    mode: Literal["classic-control"] = "classic-control"

    # Wandb / logging
    use_wandb: bool = False
    exp_name: str = "Wordle_Implementation"
    log_dir: str = "logs"
    wandb_project_name: str = "Wordle"
    wandb_entity: str = None

    # Duration of different phases
    total_timesteps: int = 500000
    num_envs: int = 8
    num_steps: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5

    # Computing advantage function
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Computing other loss functions
    clip_coef: float = 0.2
    ent_coef_start: float = 0.5
    ent_coef_end: float = 0.01
    vf_coef: float = 0.25

    def __post_init__(self):
        self.batch_size = self.num_steps * self.num_envs
        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches


def make_env(env_id: str, seed: int, idx: int, run_name: str, mode: str = "classic-control"):
    def thunk():
        env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
	
def set_global_seeds(seed):
    """Sets random seeds in several different ways (to guarantee reproducibility)"""
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True
	
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_actor_and_critic(
    envs: gym.vector.SyncVectorEnv,
    mode: Literal["classic-control", "atari", "mujoco"] = "classic-control",
) -> Tuple[nn.Module, nn.Module]:
    '''
    Returns (actor, critic), the networks used for PPO, in one of 3 different modes.
    '''
    assert mode in ["classic-control", "atari", "mujoco"]

    obs_space = envs.single_observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        num_obs = sum(np.prod(space.shape) for space in obs_space.spaces.values())
    else:
        obs_shape = obs_space.shape
        num_obs = np.array(obs_shape).prod()

    num_actions = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else np.array(envs.single_action_space.shape).prod()
    )

    if mode == "classic-control":
        actor, critic = get_actor_and_critic_classic(num_obs, num_actions)
    return actor.to(device), critic.to(device)


def get_actor_and_critic_classic(num_obs: int, num_actions: int, word_embedding_dim: int = 130):
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 1), std=1.0)
    )

    actor_base = nn.Sequential(
        layer_init(nn.Linear(num_obs, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, word_embedding_dim), std=0.01)
    )

    return actor_base, critic
	
class LearnableWordDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 130):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, state_embedding: t.Tensor) -> t.Tensor:
        logits = state_embedding @ self.word_embeddings.weight.T 
        return logits

@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    '''
    T = values.shape[0]
    next_values = t.concat([values[1:], next_value.unsqueeze(0)])
    next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
    advantages = t.zeros_like(deltas)
    advantages[-1] = deltas[-1]
    for s in reversed(range(1, T)):
        advantages[s-1] = deltas[s-1] + gamma * gae_lambda * (1.0 - dones[s]) * advantages[s]
    return advantages
	
def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''
    Return a list of length num_minibatches = (batch_size // minibatch_size), where each element is an
    array of indexes into the batch. Each index should appear exactly once.

    To relate this to the diagram above: if we flatten the non-shuffled experiences into:

        [1,1,1,1,2,2,2,2,3,3,3,3]

    then the output of this function could be the following list of arrays:

        [array([0,5,4,3]), array([11,6,7,8]), array([1,2,9,10])]

    which would give us the minibatches seen in the first row of the diagram above:

        [array([1,2,2,1]), array([3,2,2,3]), array([1,1,3,3])]
    '''
    assert batch_size % minibatch_size == 0
    indices = rng.permutation(batch_size)
    indices = einops.rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)
	
def to_numpy(arr: Union[np.ndarray, Tensor]):
    '''
    Converts a (possibly cuda and non-detached) tensor to numpy array.
    '''
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


@dataclass
class ReplayMinibatch:
    '''
    Samples from the replay memory, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_{t+1})
    '''
    observations: Tensor # shape [minibatch_size, *observation_shape]
    actions: Tensor # shape [minibatch_size, *action_shape]
    logprobs: Tensor # shape [minibatch_size,]
    advantages: Tensor # shape [minibatch_size,]
    returns: Tensor # shape [minibatch_size,]
    dones: Tensor # shape [minibatch_size,]


class ReplayMemory:
    '''
    Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
    '''
    rng: Generator

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        self.action_shape = envs.single_action_space.shape

        # Check if observations are dict-based
        self.is_dict_obs = isinstance(envs.single_observation_space, gym.spaces.Dict)
        if self.is_dict_obs:
            self.obs_keys = envs.single_observation_space.spaces.keys()
            self.obs_shape = {
                k: envs.single_observation_space.spaces[k].shape
                for k in self.obs_keys
            }

        self.reset_memory()


    def reset_memory(self):
        if self.is_dict_obs:
            self.observations = {
                k: np.empty((0, self.num_envs, *self.obs_shape[k]), dtype=np.float32)
                for k in self.obs_keys
            }
        else:
            self.observations = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)

        self.actions = np.empty((0, self.num_envs, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
        self.values = np.empty((0, self.num_envs), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
        self.dones = np.empty((0, self.num_envs), dtype=bool)

    def add(self, obs, actions, logprobs, values, rewards, dones) -> None:
      assert actions.shape == (self.num_envs, *self.action_shape)
      assert logprobs.shape == (self.num_envs,)
      assert values.shape == (self.num_envs,)
      assert dones.shape == (self.num_envs,)
      assert rewards.shape == (self.num_envs,)

      if isinstance(obs, dict):
          self.is_dict_obs = True
          for k in self.obs_keys:
              obs_k = obs[k]
              expected_shape = (self.num_envs, *self.obs_shape[k])
              assert obs_k.shape == expected_shape, f"Key '{k}' has shape {obs_k.shape}, expected {expected_shape}"
              self.observations[k] = np.concatenate((self.observations[k], to_numpy(obs_k[None, :])))
      else:
          self.is_dict_obs = False
          expected_shape = (self.num_envs, *self.obs_shape)
          assert obs.shape == expected_shape, f"Obs has shape {obs.shape}, expected {expected_shape}"
          self.observations = np.concatenate((self.observations, to_numpy(obs[None, :])))

      self.actions = np.concatenate((self.actions, to_numpy(actions[None, :])))
      self.logprobs = np.concatenate((self.logprobs, to_numpy(logprobs[None, :])))
      self.values = np.concatenate((self.values, to_numpy(values[None, :])))
      self.rewards = np.concatenate((self.rewards, to_numpy(rewards[None, :])))
      self.dones = np.concatenate((self.dones, to_numpy(dones[None, :])))



    def get_minibatches(self, next_value: t.Tensor, next_done: t.Tensor) -> List[ReplayMinibatch]:
        minibatches = []

        if self.is_dict_obs:
            obs = {
                k: preprocess_obs(t.from_numpy(self.observations[k]).to(device))
                for k in self.obs_keys
            }
        else:
            obs = preprocess_obs(t.from_numpy(self.observations).to(device))

        actions = t.from_numpy(self.actions).to(device)
        logprobs = t.from_numpy(self.logprobs).to(device)
        values = t.from_numpy(self.values).to(device)
        rewards = t.from_numpy(self.rewards).to(device)
        dones = t.from_numpy(self.dones).to(device)

        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones.float(),
            self.args.gamma, self.args.gae_lambda
        )
        returns = advantages + values

        # Flatten dimensions
        if self.is_dict_obs:
            obs_flat = {k: v.flatten(0, 1) for k, v in obs.items()}
        else:
            obs_flat = obs.flatten(0, 1)

        actions_flat = actions.flatten(0, 1)
        logprobs_flat = logprobs.flatten(0, 1)
        advantages_flat = advantages.flatten(0, 1)
        returns_flat = returns.flatten(0, 1)
        dones_flat = dones.flatten(0, 1)

        # Index and create minibatches
        for _ in range(self.args.batches_per_learning_phase):
            for indices in minibatch_indexes(self.rng, self.args.batch_size, self.args.minibatch_size):
                if self.is_dict_obs:
                    mb_obs = {k: v[indices] for k, v in obs_flat.items()}
                else:
                    mb_obs = obs_flat[indices]

                minibatches.append(ReplayMinibatch(
                    observations=mb_obs,
                    actions=actions_flat[indices],
                    logprobs=logprobs_flat[indices],
                    advantages=advantages_flat[indices],
                    returns=returns_flat[indices],
                    dones=dones_flat[indices],
                ))

        self.reset_memory()
        return minibatches
		
def build_vocab_matrix(words: list[str]) -> np.ndarray:
    vocab_size = len(words)
    word_len = 5
    vocab_matrix = np.zeros((vocab_size, 130), dtype=np.float32)  # 26 * 5 = 130
    for i, word in enumerate(words):
        for j, c in enumerate(word):
            vocab_matrix[i, j * 26 + (ord(c) - ord('A'))] = 1.0
    return vocab_matrix
	
class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.args = args
        self.envs = envs

        # Keep track of global number of steps taken by agent
        self.step = 0

        # Get actor and critic networks
        self.actor, self.critic = get_actor_and_critic(envs, mode=args.mode)
        raw_env = get_inner_env(self.envs.envs[0])
        vocab_matrix = build_vocab_matrix(raw_env.words)
        vocab_size = len(raw_env.words)
        self.decoder = LearnableWordDecoder(vocab_size=vocab_size, embedding_dim=130).to(device)

        obs, _ = envs.reset()  # Get observation and info from reset
        if isinstance(obs, dict):  # Check if observation is a dictionary
            self.next_obs = {k: t.tensor(v).to(device, dtype=t.float) for k, v in obs.items()}  # Convert each value to a tensor
        else:
            self.next_obs = t.tensor(obs).to(device, dtype=t.float)  # Handle non-dictionary observations
        self.next_done = t.zeros(envs.num_envs).to(device, dtype=t.float)

        # Create our replay memory
        self.memory = ReplayMemory(args, envs)


    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        '''
        raw_obs = self.next_obs
        obs = preprocess_obs(raw_obs)
        dones = self.next_done

        with t.inference_mode():
          state_embedding = self.actor(obs)
          logits = self.decoder(state_embedding)
          if isinstance(self.next_obs, dict): 
		guessed = t.as_tensor(self.next_obs["guessed"], device=logits.device).bool()
		logits = logits.masked_fill(guessed, -1e9)
        probs = Categorical(logits=logits)
        actions = probs.sample()

        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions.cpu().numpy())
        next_dones = terminated | truncated

        logprobs = probs.log_prob(actions)
        with t.inference_mode():
            values = self.critic(obs).flatten()

        self.memory.add(raw_obs, actions, logprobs, values, rewards, dones)

        self.next_obs = next_obs
        self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
        self.step += self.envs.num_envs

        return infos



    def get_minibatches(self) -> list[ReplayMinibatch]:
        '''
        Gets minibatches from the replay memory.
        '''
        with t.inference_mode():
            next_value = self.critic(preprocess_obs(self.next_obs)).flatten()
        return self.memory.get_minibatches(next_value, self.next_done)
		
def calc_clipped_surrogate_objective(
    probs: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8
) -> Float[Tensor, ""]:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    '''
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    logits_diff = probs.log_prob(mb_action) - mb_logprobs
    r_theta = t.exp(logits_diff)
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)
    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()

def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float
) -> Float[Tensor, ""]:
    '''Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape
    return vf_coef * (values - mb_returns).pow(2).mean()
	
def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()
	
class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_training_steps: int, ent_coef_start: float, ent_coef_end: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_training_steps
        assert frac <= 1

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

    def current_entropy_coef(self) -> float:
        frac = self.n_step_calls / self.total_training_steps
        return self.ent_coef_start + frac * (self.ent_coef_end - self.ent_coef_start)
		
def make_optimizer(
    agent: PPOAgent,
    total_training_steps: int,
    initial_lr: float,
    end_lr: float,
    extra_params=[],
    ent_coef_start: float = 0.5,
    ent_coef_end: float = 0.01
) -> tuple[optim.Adam, PPOScheduler]:
    """
    Creates an Adam optimizer and PPOScheduler with learning rate and entropy coefficient annealing.
    """
    params = list(agent.parameters()) + extra_params
    optimizer = optim.Adam(params, lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(
        optimizer=optimizer,
        initial_lr=initial_lr,
        end_lr=end_lr,
        total_training_steps=total_training_steps,
        ent_coef_start=ent_coef_start,
        ent_coef_end=ent_coef_end
    )
    return optimizer, scheduler
	
class PPOTrainer:

    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, self.run_name, args.mode) for i in range(args.num_envs)])
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(
            self.agent,
            total_training_steps=self.args.total_training_steps,
            initial_lr=self.args.learning_rate,
            end_lr=0.0,
            extra_params=list(self.agent.decoder.parameters()),
            ent_coef_start=self.args.ent_coef_start,
            ent_coef_end=self.args.ent_coef_end
        )




    def rollout_phase(self) -> Optional[int]:
        '''
        This function populates the memory with a new set of experiences, using `self.agent.play_step`
        to step through the environment. It also returns the episode length of the most recently terminated
        episode (used in the progress bar readout).
        '''
        last_episode_len = None
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()
            for info in infos:
                if isinstance(info, dict) and "episode" in info: # Changed from info.keys() to info
                  last_episode_len = info["episode"]["l"]
                  last_episode_return = info["episode"]["r"]
                  if self.args.use_wandb: wandb.log({
                      "episode_length": last_episode_len,
                      "episode_return": last_episode_return
                  }, step=self.agent.step)
        return last_episode_len


    def learning_phase(self) -> None:
        '''
        This function does the following:

            - Generates minibatches from memory
            - Calculates the objective function, and takes an optimization step based on it
            - Clips the gradients (see detail #11)
            - Steps the learning rate scheduler
        '''
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective_fn = self.compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        '''
        Handles learning phase for a single minibatch. Returns objective function to be maximized.
        '''
        # handles both dict and tensor
        obs = preprocess_obs(minibatch.observations)

        state_emb = self.agent.actor(obs)         # [B, 130]
        logits = self.agent.decoder(state_emb)    # [B, vocab_size]
        probs = Categorical(logits=logits)
        values = self.agent.critic(obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            probs, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef
        )
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_coef = self.scheduler.current_entropy_coef()
        entropy_bonus = calc_entropy_bonus(probs, entropy_coef)


        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            newlogprob = probs.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb:
            wandb.log(dict(
                total_steps = self.agent.step,
                values = values.mean().item(),
                learning_rate = self.scheduler.optimizer.param_groups[0]["lr"],
                value_loss = value_loss.item(),
                clipped_surrogate_objective = clipped_surrogate_objective.item(),
                entropy = entropy_bonus.item(),
                approx_kl = approx_kl,
                clipfrac = np.mean(clipfracs)
            ), step=self.agent.step)

        return total_objective_function

    def train(self) -> None:

        if args.use_wandb: wandb.init(
            project=self.args.wandb_project_name,
            entity=self.args.wandb_entity,
            name=self.run_name,
            monitor_gym=False
        )

        progress_bar = tqdm(range(self.args.total_phases))

        for epoch in progress_bar:

            last_episode_len = self.rollout_phase()
            if last_episode_len is not None:
                progress_bar.set_description(f"Epoch {epoch:02}, Episode length: {last_episode_len}")

            self.learning_phase()

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()
			
register_all_wordle_envs()

def evaluate_agent(agent, env, words, num_episodes=5):
    successes = 0
    total_guesses = 0

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        print(f"\n Episode {i+1}")

        while not done:
            obs_t = preprocess_obs(obs).unsqueeze(0)  
            with t.no_grad():
                state_emb = agent.actor(obs_t)
                logits = agent.decoder(state_emb)

                if isinstance(obs, dict) and "guessed" in obs:
                    guessed = obs["guessed"]
		    guessed = t.as_tensor(guessed, device=logits.device).bool()
		    logits[0, guessed] = -1e9


                action = t.argmax(logits, dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Guess {step+1}: {info['guess']} — Mask: {info['mask']}")
            step += 1

        print("Success!" if reward > 0 else "Failed :(")
        successes += int(reward > 0)
        total_guesses += step

    print(f"\n Success rate: {successes}/{num_episodes} ({100 * successes / num_episodes:.2f}%)")
    print(f" Avg guesses per game: {total_guesses / num_episodes:.2f}")
	
def load_partial_state_dict(model: t.nn.Module, checkpoint_path: str, verbose: bool = True):
    """Loads weights from a checkpoint, ignoring any size mismatches (e.g., decoder layers)."""
    checkpoint = t.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    filtered_dict = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    if verbose:
        loaded_keys = list(filtered_dict.keys())
        skipped_keys = [k for k in checkpoint if k not in filtered_dict]
        print(f"Loaded {len(loaded_keys)} matching keys from checkpoint.")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys due to size mismatch:")
            for key in skipped_keys:
                print(f"  - {key}")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

def build_mask_table(words):
    V = len(words)
    mask_ids = np.empty((V, V), dtype=np.int16)
    for i in range(V):
        for j in range(V):
            m = get_mask(words[i], words[j])     
            code = m[0] + 3*(m[1] + 3*(m[2] + 3*(m[3] + 3*m[4])))
            mask_ids[i, j] = code
    return mask_ids

def expected_entropy_drop_fast(env, guess_idx):
    """
    Fast O(|C|) expected log‑|C| drop using precomputed mask_table.
    """
    C_idxs = np.array([word2idx[w] for w in env.remaining_candidates], dtype=int)
    old_log = math.log(len(C_idxs))

    row = mask_table[guess_idx, C_idxs]      
    vals, counts = np.unique(row, return_counts=True)
    return ((old_log - np.log(counts)) * (counts / len(C_idxs))).sum()

def expert_action(env):
    """
    Scan only remaining candidates; pick index with highest expected drop.
    """
    best_idx, best_score = None, -1.0
    for w in env.remaining_candidates:
        idx = word2idx[w]
        score = expected_entropy_drop_fast(env, idx)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx

def behavioral_clone(actor, decoder, 
                     states, actions, 
                     epochs=5, batch_size=64, lr=1e-3, device='cuda'):
   
    actor.train(); decoder.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(decoder.parameters()), 
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(states.to(device), actions.to(device))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for s_batch, a_batch in loader:
            logits = decoder(actor(s_batch))      
            loss   = criterion(logits, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
        avg = total_loss / len(dataset)
        print(f"[BC] Epoch {epoch}/{epochs} — loss {avg:.4f}")

args = PPOArgs(
    env_id="Wordle1000-v0",
    num_envs=8,
    total_timesteps=10_000_000,
    num_steps=128,
    learning_rate=1e-5,
    use_wandb=True,
    seed=1
)
trainer = PPOTrainer(args)
env0 = get_inner_env(trainer.envs.envs[0])  

raw_env = get_inner_env(trainer.envs.envs[0])
words   = raw_env.words
word2idx = {w:i for i,w in enumerate(words)}
mask_table = build_mask_table(words)

bc_states, bc_actions = generate_expert_dataset(
    env0, preprocess_obs, num_episodes=200
)

behavioral_clone(
    trainer.agent.actor, 
    trainer.agent.decoder,
    bc_states, bc_actions,
    epochs=5, batch_size=128, lr=1e-3,
    device=device
)


print("Starting PPO finetuning…")
trainer.train()

CURRICULUM_STAGES = [
    ("Wordle100-v0", 5_000_000),
    ("Wordle1000-v0, 5_000_000),
    ("WordleFull-v0, 5_000_000) 
]

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

total_steps_so_far = 0

for idx, (env_id, stage_steps) in enumerate(CURRICULUM_STAGES):
    print(f"\n Stage {idx+1}/{len(CURRICULUM_STAGES)}: {env_id} for {stage_steps} steps")

    args = PPOArgs(
        env_id=env_id,
        num_envs=8,
        total_timesteps=stage_steps,
        num_steps=128,
        learning_rate=1e-5,
        exp_name=f"wordle-curriculum-stage{idx+1}",
        use_wandb=True,
        seed=1
    )

    trainer = PPOTrainer(args)

    if idx > 0:
        ckpt_path = os.path.join(SAVE_DIR, f"stage{idx}_agent.pt")
        state_dict = t.load(ckpt_path)
        load_partial_state_dict(trainer.agent, ckpt_path)

    trainer.train()

    ckpt_path = os.path.join(SAVE_DIR, f"stage{idx+1}_agent.pt")
    t.save(trainer.agent.state_dict(), ckpt_path)
    print(f" Saved model to {ckpt_path}")

    print(f"\n Evaluating {env_id} agent after {stage_steps} steps")
    raw_env = get_inner_env(gym.make(env_id))
    eval_env = raw_env.__class__()  
    evaluate_agent(trainer.agent, eval_env, raw_env.words, num_episodes=10)

    total_steps_so_far += stage_steps
    print(f" Total steps so far: {total_steps_so_far}")
