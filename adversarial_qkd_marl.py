import os
import random
import math
import json
import csv
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import defaultdict

import gymnasium as gym
import numpy as np

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.algorithms.maddpg import MADDPGConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1) QUANTUM QKD PHYSICS HELPER FUNCTIONS
# ============================================================

def shannon_binary_entropy(qber: float) -> float:
    """
    H2(q) = -q log2(q) - (1-q) log2(1-q), clipped for stability.
    """
    q = float(np.clip(qber, 1e-8, 1.0 - 1e-8))
    return float(-(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q)))


def compute_skr(qber: float, f_efficiency: float = 1.16) -> float:
    """
    Secret Key Rate:
      SKR = max(0.0, 1.0 - (1.0 + f_efficiency) * H2(QBER))
    """
    if qber < 0.0 or qber > 1.0:
        qber = float(np.clip(qber, 0.0, 1.0))
    h2 = shannon_binary_entropy(qber)
    return max(0.0, 1.0 - (1.0 + f_efficiency) * h2)


# ============================================================
# 2) ADAPTIVE THREAT MODEL
# ============================================================

class AdaptiveEavesdropper:
    """
    Game-theoretic adversary with adaptive intensity scaling.
    - Attack active in 30% of episodes
    - If mean defense threshold is high, the adversary halves attack intensity
    """

    def __init__(self, attack_probability: float = 0.30, attack_intensity: float = 1.0):
        self.attack_probability = float(np.clip(attack_probability, 0.0, 1.0))
        self.attack_intensity = float(attack_intensity)

    def should_attack(self, episode_index: int, num_episodes: int, rng: Optional[np.random.Generator] = None) -> bool:
        if rng is None:
            rng = np.random.default_rng()
        if num_episodes <= 0:
            return False
        active_window = int(np.ceil(num_episodes * self.attack_probability))
        if active_window <= 0:
            return False
        return episode_index % max(1, num_episodes // max(1, active_window)) == 0 and (
            rng.random() < self.attack_probability
        )

    def compute_attack_intensity(self, avg_defense_threshold: float) -> float:
        """
        If average defense threshold is high, adversary evades by reducing attack intensity by half.
        """
        if avg_defense_threshold >= 0.7:
            return self.attack_intensity * 0.5
        return self.attack_intensity


# ============================================================
# 3) CENTRALIZED CRITIC MODEL FOR MAPPO
# ============================================================

class CentralizedCriticModel(TorchModelV2):
    """
    Centralized critic model for MAPPO.
    Observes all agents' observations and outputs a shared value function.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        # Actor network (local policy)
        self.actor_net = nn.Sequential(
            nn.Linear(int(np.prod(obs_space.shape)), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

        # Critic network (shared for all agents in centralized setup)
        # In MAPPO, the critic sees a concatenated observation of all agents
        self.critic_net = nn.Sequential(
            nn.Linear(int(np.prod(obs_space.shape)), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        logits = self.actor_net(obs)
        self._value_out = self.critic_net(obs)
        return logits, state

    def value_function(self):
        return self._value_out.squeeze(-1)


# ============================================================
# 4) VALUE FACTORIZATION MIXER FOR QMIX
# ============================================================

class QMixMixer(nn.Module):
    """
    QMix mixer network: computes joint Q-value from individual Q-values.
    """

    def __init__(self, num_agents: int, qmix_hidden_dim: int = 32, state_dim: int = 3):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Weights network (for each agent Q-value)
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, num_agents),
        )

        # Bias network (global bias)
        self.bias_net = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, 1),
        )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        q_values: shape (batch_size, num_agents)
        state: shape (batch_size, state_dim)
        returns: joint_q_value (batch_size, 1)
        """
        weights = self.weight_net(state)  # (batch_size, num_agents)
        weights = torch.abs(weights)  # Ensure positive weights
        bias = self.bias_net(state)  # (batch_size, 1)

        weighted_q = (q_values * weights).sum(dim=1, keepdim=True)
        joint_q = weighted_q + bias
        return joint_q


class VDNMixer(nn.Module):
    """
    VDN (Value Decomposition Networks) mixer: simple sum of individual Q-values.
    No learnable parameters.
    """

    def __init__(self, num_agents: int, **kwargs):
        super().__init__()
        self.num_agents = num_agents

    def forward(self, q_values: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q_values: shape (batch_size, num_agents)
        returns: joint_q_value (batch_size, 1)
        """
        joint_q = q_values.sum(dim=1, keepdim=True)
        return joint_q


# ============================================================
# 5) ENVIRONMENT: GYMNASIUM + MULTI-AGENT WRAPPER
# ============================================================

class AdversarialQKDEnv(MultiAgentEnv):
    """
    Multi-agent environment for distributed QKD network defense.
    Each node agent controls a scalar defense threshold.
    Observations: [Current_QBER, Calculated_SKR, Network_Detection_State]
    Action: continuous Box threshold per node
    """

    metadata = {"render_modes": [], "name": "AdversarialQKDEnv-v0"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}
        self.num_agents = int(cfg.get("num_agents", 5))
        self.max_steps = int(cfg.get("max_steps", 20))
        self.episode_len = self.max_steps
        self.qber_base = float(cfg.get("qber_base", 0.05))
        self.qber_std = float(cfg.get("qber_std", 0.02))
        self.attack_severity = float(cfg.get("attack_severity", 0.18))
        self.defense_threshold_range = (0.0, 1.0)

        self.rng = np.random.default_rng(seed=cfg.get("seed", 42))
        self.adversary = AdaptiveEavesdropper(
            attack_probability=cfg.get("attack_probability", 0.30),
            attack_intensity=cfg.get("attack_intensity", 1.0),
        )

        self.current_step = 0
        self.current_episode = 0
        self.episode_active_attack = False
        self.current_attack_intensity = 1.0
        self.total_throughput = 0.0
        self.total_security_penalty = 0.0
        self.global_history = {
            "mean_qber": 0.0,
            "mean_skr": 0.0,
            "detection_f1_score": 0.0,
            "detection_auc_roc": 0.0,
        }

        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        # Observation: [Current_QBER, Calculated_SKR, Network_Detection_State]
        obs_dim = 3
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Each agent chooses a continuous defense threshold [0,1]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_metrics()

    def _reset_metrics(self):
        self.current_step = 0
        self.current_episode += 1
        self.episode_active_attack = False
        self.current_attack_intensity = 1.0
        self.total_throughput = 0.0
        self.total_security_penalty = 0.0

    def _compute_classification_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray):
        if len(true_labels) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc_roc": 0.0,
            }

        try:
            precision = precision_score(true_labels, pred_labels, average="binary", zero_division=0)
        except (ValueError, IndexError):
            precision = 0.0

        try:
            recall = recall_score(true_labels, pred_labels, average="binary", zero_division=0)
        except (ValueError, IndexError):
            recall = 0.0

        try:
            f1 = f1_score(true_labels, pred_labels, average="binary", zero_division=0)
        except (ValueError, IndexError):
            f1 = 0.0

        try:
            auc_roc = roc_auc_score(true_labels, pred_labels)
        except (ValueError, IndexError):
            auc_roc = 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc_roc),
        }

    def _generate_observation(self, qber_value: float, detect_state: float) -> np.ndarray:
        skr = compute_skr(qber_value, f_efficiency=1.16)
        obs = np.array([qber_value, skr, detect_state], dtype=np.float32)
        return obs

    def _network_state(self):
        detection_state = float(np.clip(np.mean([0.0 for _ in range(self.num_agents)]), 0.0, 1.0))
        return detection_state

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self._reset_metrics()

        active_attack_probability = self.adversary.attack_probability
        self.episode_active_attack = bool(self.rng.random() < active_attack_probability)

        if self.episode_active_attack:
            attack_scale = self.rng.uniform(0.15, 0.45)
            self.current_attack_intensity = float(attack_scale * self.attack_severity)
        else:
            self.current_attack_intensity = 0.0

        qber_base = self.qber_base + self.rng.normal(0.0, self.qber_std)
        qber_value = float(np.clip(qber_base, 0.0, 0.95))
        detect_state = float(self.rng.random() * 0.5 + (0.25 if self.episode_active_attack else 0.1))
        obs = {agent_id: self._generate_observation(qber_value, detect_state) for agent_id in self.agent_ids}
        return obs, {}

    def _step_attack(self, action_dict: Dict[str, np.ndarray], qber_value: float):
        attack_probability = self.current_attack_intensity
        avg_thresh = float(np.mean([float(action_dict[agent_id][0]) for agent_id in self.agent_ids]))
        attack_intensity = self.adversary.compute_attack_intensity(avg_thresh) * attack_probability
        if attack_intensity > 0:
            self.episode_active_attack = True

        qber_increase = attack_intensity * np.clip(self.rng.normal(0.13, 0.06), 0.0, 0.8)
        new_qber = float(np.clip(qber_value + qber_increase, 0.0, 1.0))
        return new_qber, attack_intensity

    def step(self, action_dict: Dict[str, np.ndarray]):
        if self.current_step >= self.max_steps:
            return self._done_state()

        action_dict = {
            agent_id: np.clip(np.asarray(action_dict[agent_id], dtype=np.float32), 0.0, 1.0)
            for agent_id in self.agent_ids
        }

        qber_value = self.qber_base + self.rng.normal(0.0, self.qber_std)
        qber_value = float(np.clip(qber_value, 0.0, 1.0))

        qber_after_attack, attack_intensity = self._step_attack(action_dict, qber_value)
        detect_state = float(np.clip(0.1 + 0.7 * (qber_after_attack / (1.0 + 1e-6)), 0.0, 1.0))
        skr = compute_skr(qber_after_attack, f_efficiency=1.16)

        alpha, beta, gamma = 2.0, 5.0, 3.0
        penalty_total = 0.0
        rewards = {}

        true_detect_labels = []
        pred_detect_labels = []
        miss_count = 0

        for agent_id in self.agent_ids:
            defense_threshold = float(action_dict[agent_id][0])
            attack_event = (self.episode_active_attack and attack_intensity > 0.0)
            detected = attack_event and (defense_threshold >= max(0.35, qber_after_attack))
            false_negative = attack_event and (not detected)
            if false_negative:
                miss_count += 1
                penalty_total += 4.0 * gamma * (1.0 + attack_intensity)
            else:
                penalty_total += gamma * (0.25 * max(0.0, 1.0 - qber_after_attack))

            individual_penalty = gamma * (1.0 if false_negative else 0.25) * (1.0 + attack_intensity)
            if false_negative:
                individual_penalty *= 4.0

            reward = alpha * skr - beta * qber_after_attack - individual_penalty
            rewards[agent_id] = float(reward)

            true_detect_labels.append(1 if attack_event else 0)
            pred_detect_labels.append(1 if detected else 0)

        true_labels = np.asarray(true_detect_labels, dtype=int)
        pred_labels = np.asarray(pred_detect_labels, dtype=int)

        if np.unique(true_labels).size > 1:
            metrics = self._compute_classification_metrics(true_labels, pred_labels)
        else:
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc_roc": 0.0,
            }

        self.global_history["mean_qber"] = (
            (self.global_history["mean_qber"] * self.current_step + qber_after_attack) / (self.current_step + 1)
        )
        self.global_history["mean_skr"] = (
            (self.global_history["mean_skr"] * self.current_step + skr) / (self.current_step + 1)
        )
        self.global_history["detection_f1_score"] = (
            (self.global_history["detection_f1_score"] * self.current_step + metrics["f1"]) / (self.current_step + 1)
        )
        self.global_history["detection_auc_roc"] = (
            (self.global_history["detection_auc_roc"] * self.current_step + metrics["auc_roc"]) / (self.current_step + 1)
        )

        obs = {
            agent_id: self._generate_observation(qber_after_attack, detect_state)
            for agent_id in self.agent_ids
        }

        terminated = {agent_id: False for agent_id in self.agent_ids}
        truncated = {agent_id: False for agent_id in self.agent_ids}
        dones = {"__all__": False}

        self.current_step += 1
        if self.current_step >= self.max_steps:
            dones["__all__"] = True
            terminated = {agent_id: True for agent_id in self.agent_ids}
            truncated = {agent_id: True for agent_id in self.agent_ids}

        return obs, rewards, terminated, truncated, {
            "qber": qber_after_attack,
            "skr": skr,
            "detect_state": detect_state,
            "attack_intensity": attack_intensity,
            "episode_active_attack": self.episode_active_attack,
            "metrics": metrics,
            "history": self.global_history,
        }

    def _done_state(self):
        obs = {agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent_id in self.agent_ids}
        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        terminated = {agent_id: True for agent_id in self.agent_ids}
        truncated = {agent_id: False for agent_id in self.agent_ids}
        dones = {"__all__": True}
        return obs, rewards, terminated, truncated, {"done": True}


# ============================================================
# 6) RAY RLLIB CALLBACKS PIPELINE
# ============================================================

class QKDMarlMetricsCallback(DefaultCallbacks):
    """
    Logs telemetry across steps and aggregates metrics at episode end.
    Compatible with RLlib 2.x API.
    """

    def __init__(self):
        super().__init__()
        self.episode_qber = []
        self.episode_skr = []
        self.episode_f1 = []
        self.episode_auc = []
        self.episode_rewards = defaultdict(list)
        self.global_history = {
            "mean_qber": [],
            "mean_skr": [],
            "detection_f1_score": [],
            "detection_auc_roc": [],
        }

    def on_episode_step(
        self,
        *,
        algorithm,
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        """Collect metrics during episode step."""
        try:
            info = episode.last_info_for_agent
            if info and isinstance(info, dict):
                if "qber" in info:
                    self.episode_qber.append(float(info["qber"]))
                if "skr" in info:
                    self.episode_skr.append(float(info["skr"]))
                if "metrics" in info:
                    metrics = info["metrics"]
                    if isinstance(metrics, dict):
                        if "f1" in metrics:
                            self.episode_f1.append(float(metrics["f1"]))
                        if "auc_roc" in metrics:
                            self.episode_auc.append(float(metrics["auc_roc"]))
        except Exception as e:
            pass

    def on_episode_end(
        self,
        *,
        algorithm,
        episode: Episode,
        **kwargs
    ) -> None:
        """Aggregate and log metrics at episode end."""
        qber_mean = float(np.mean(self.episode_qber)) if len(self.episode_qber) > 0 else 0.0
        skr_mean = float(np.mean(self.episode_skr)) if len(self.episode_skr) > 0 else 0.0
        f1_mean = float(np.mean(self.episode_f1)) if len(self.episode_f1) > 0 else 0.0
        auc_mean = float(np.mean(self.episode_auc)) if len(self.episode_auc) > 0 else 0.0

        self.global_history["mean_qber"].append(qber_mean)
        self.global_history["mean_skr"].append(skr_mean)
        self.global_history["detection_f1_score"].append(f1_mean)
        self.global_history["detection_auc_roc"].append(auc_mean)

        episode.custom_metrics["mean_qber"] = qber_mean
        episode.custom_metrics["mean_skr"] = skr_mean
        episode.custom_metrics["detection_f1_score"] = f1_mean
        episode.custom_metrics["detection_auc_roc"] = auc_mean
        episode.custom_metrics["f1_score"] = f1_mean
        episode.custom_metrics["auc_roc"] = auc_mean

        self.episode_qber = []
        self.episode_skr = []
        self.episode_f1 = []
        self.episode_auc = []


# ============================================================
# 7) MARL CONFIGURATION ARCHITECTURE (RLlib 2.x API)
# ============================================================

def get_policy_mapping_fn(agent_id: str, episode, worker, **kwargs) -> str:
    """Simple policy mapping: one policy per agent."""
    return agent_id


def get_marl_config(algorithm_name: str, env_config: Optional[Dict[str, Any]] = None, num_rollout_workers: int = 1):
    """
    Returns RLlib AlgorithmConfig depending on algorithm string.
    Strict RLlib 2.x API compliance.
    Supported: "IPPO", "MAPPO", "MADDPG", "QMIX", "VDN"
    """
    env_config = env_config or {}
    env = AdversarialQKDEnv(config=env_config)
    algo_name = algorithm_name.upper()

    # Policy specs for all agents
    policies = {
        agent_id: PolicySpec(
            policy_class=None,
            observation_space=env.observation_space,
            action_space=env.action_space,
            config={},
        )
        for agent_id in env.agent_ids
    }

    if algo_name == "IPPO":
        """
        Independent PPO: Fully decentralized, each agent has its own independent policy.
        No value sharing or centralized critic.
        """
        cfg = (
            PPOConfig()
            .environment(env=AdversarialQKDEnv, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=num_rollout_workers, num_envs_per_worker=1)
            .training(
                gamma=0.99,
                lr=3e-4,
                kl_coeff=0.2,
                clip_param=0.2,
                train_batch_size=256,
                sgd_minibatch_size=64,
                num_sgd_iter=10,
                model={
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "relu",
                    "vf_share_layers": False,
                }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=get_policy_mapping_fn,
                policies_to_train=list(env.agent_ids),
            )
            .callbacks(QKDMarlMetricsCallback)
            .debugging(log_level="INFO")
        )
        return cfg

    elif algo_name == "MAPPO":
        """
        Multi-Agent PPO with Centralized Critic (MAPPO).
        Uses a shared value function during training but decentralized execution.
        Explicit centralized critic architecture.
        """
        # Register centralized critic model
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

        cfg = (
            PPOConfig()
            .environment(env=AdversarialQKDEnv, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=num_rollout_workers, num_envs_per_worker=1)
            .training(
                gamma=0.99,
                lr=3e-4,
                kl_coeff=0.2,
                clip_param=0.2,
                train_batch_size=256,
                sgd_minibatch_size=64,
                num_sgd_iter=10,
                model={
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "relu",
                    "vf_share_layers": False,
                    "custom_model": "cc_model",
                    "max_seq_len": 20,
                }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=get_policy_mapping_fn,
                policies_to_train=list(env.agent_ids),
            )
            .callbacks(QKDMarlMetricsCallback)
            .debugging(log_level="INFO")
        )
        return cfg

    elif algo_name == "MADDPG":
        """
        Multi-Agent DDPG (MADDPG).
        Centralized training with decentralized execution.
        Each agent has actor and critic networks.
        """
        cfg = (
            DDPGConfig()
            .environment(env=AdversarialQKDEnv, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=num_rollout_workers, num_envs_per_worker=1)
            .training(
                gamma=0.99,
                lr=1e-3,
                actor_lr=1e-3,
                critic_lr=1e-3,
                train_batch_size=256,
                tau=0.005,
                target_network_update_freq=1,
                model={
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "relu",
                    "post_fcnet_hiddens": [64],
                    "post_fcnet_activation": "relu",
                }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=get_policy_mapping_fn,
                policies_to_train=list(env.agent_ids),
            )
            .callbacks(QKDMarlMetricsCallback)
            .debugging(log_level="INFO")
        )
        return cfg

    elif algo_name == "QMIX":
        """
        QMIX: Value Factorization Network.
        Mixes individual Q-values using a learned mixer network.
        The mixer learns how to combine agent Q-values into a joint Q-value.
        """
        cfg = (
            QMixConfig()
            .environment(env=AdversarialQKDEnv, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=num_rollout_workers, num_envs_per_worker=1)
            .training(
                gamma=0.99,
                lr=5e-4,
                train_batch_size=256,
                learning_starts=1000,
                epsilon_timesteps=10000,
                epsilon_start=1.0,
                epsilon_final=0.02,
                target_network_update_freq=200,
                double_q=False,
                dueling_q_networks=False,
                model={
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "relu",
                    "use_lstm": False,
                    "max_seq_len": 20,
                }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=get_policy_mapping_fn,
                policies_to_train=list(env.agent_ids),
            )
            .callbacks(QKDMarlMetricsCallback)
            .debugging(log_level="INFO")
        )
        return cfg

    elif algo_name == "VDN":
        """
        VDN: Value Decomposition Networks.
        Simpler than QMIX: mixes individual Q-values as a simple sum.
        No learnable mixing parameters, strictly additive factorization.
        """
        cfg = (
            QMixConfig()
            .environment(env=AdversarialQKDEnv, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=num_rollout_workers, num_envs_per_worker=1)
            .training(
                gamma=0.99,
                lr=5e-4,
                train_batch_size=256,
                learning_starts=1000,
                epsilon_timesteps=10000,
                epsilon_start=1.0,
                epsilon_final=0.02,
                target_network_update_freq=200,
                double_q=False,
                dueling_q_networks=False,
                model={
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "relu",
                    "use_lstm": False,
                    "max_seq_len": 20,
                }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=get_policy_mapping_fn,
                policies_to_train=list(env.agent_ids),
            )
            .callbacks(QKDMarlMetricsCallback)
            .debugging(log_level="INFO")
        )
        # VDN uses simple summation instead of learned mixing
        cfg.training()["mixer"] = "vdn"
        return cfg

    else:
        raise ValueError(f"Unsupported MARL algorithm: {algorithm_name}")


# ============================================================
# 8) TRAINING PIPELINE & EXECUTION
# ============================================================

def run_single_training(
    algorithm_name: str,
    env_config: Dict[str, Any],
    num_iterations: int = 10,
    num_rollout_workers: int = 1,
    results_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run training for a single algorithm and return comprehensive metrics.
    """
    if results_list is None:
        results_list = []

    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    except Exception:
        pass

    config = get_marl_config(algorithm_name, env_config, num_rollout_workers=num_rollout_workers)
    trainer = config.build()

    print(f"\n{'='*80}")
    print(f"Training {algorithm_name.upper()} for {num_iterations} iterations")
    print(f"{'='*80}")

    iteration_history = []

    for iteration in range(num_iterations):
        result = trainer.train()

        # Extract metrics from RLlib result dict (RLlib 2.x format)
        episode_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0.0)
        if not episode_reward_mean:
            episode_reward_mean = result.get("episode_reward_mean", 0.0)

        episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0.0)
        if not episode_len_mean:
            episode_len_mean = result.get("episode_len_mean", 0.0)

        custom_metrics = result.get("custom_metrics", {})
        mean_qber = custom_metrics.get("mean_qber", 0.0)
        mean_skr = custom_metrics.get("mean_skr", 0.0)
        detection_f1_score = custom_metrics.get("detection_f1_score", 0.0)
        detection_auc_roc = custom_metrics.get("detection_auc_roc", 0.0)

        metrics = {
            "algorithm": algorithm_name,
            "iteration": iteration + 1,
            "episode_reward_mean": float(episode_reward_mean),
            "episode_len_mean": float(episode_len_mean),
            "mean_qber": float(mean_qber),
            "mean_skr": float(mean_skr),
            "detection_f1_score": float(detection_f1_score),
            "detection_auc_roc": float(detection_auc_roc),
        }
        iteration_history.append(metrics)
        results_list.append(metrics)

        print(
            f"Iter {iteration+1:02d}/{num_iterations:02d} | "
            f"reward={metrics['episode_reward_mean']:8.4f} | "
            f"len={metrics['episode_len_mean']:6.2f} | "
            f"qber={metrics['mean_qber']:.4f} | "
            f"skr={metrics['mean_skr']:.4f} | "
            f"f1={metrics['detection_f1_score']:.4f} | "
            f"auc={metrics['detection_auc_roc']:.4f}"
        )

    trainer.stop()
    return iteration_history


def export_results_to_csv(results: List[Dict[str, Any]], filename: str = "qkd_marl_results.csv"):
    """Export benchmark results to CSV for further analysis."""
    if not results:
        print(f"[WARNING] No results to export to {filename}")
        return

    keys = list(results[0].keys())
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] Results exported to {filename}")


def plot_benchmark_results(results: List[Dict[str, Any]], output_dir: str = "./qkd_plots"):
    """
    Generate comparison plots for all algorithms.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Organize results by algorithm
    algo_results = defaultdict(list)
    for result in results:
        algo = result["algorithm"]
        algo_results[algo].append(result)

    algorithms = sorted(algo_results.keys())
    colors = {
        "IPPO": "#1f77b4",
        "MAPPO": "#ff7f0e",
        "MADDPG": "#2ca02c",
        "QMIX": "#d62728",
        "VDN": "#9467bd",
    }

    # Plot 1: Episode Reward Mean
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        iterations = [r["iteration"] for r in algo_results[algo]]
        rewards = [r["episode_reward_mean"] for r in algo_results[algo]]
        plt.plot(iterations, rewards, marker='o', label=algo, color=colors.get(algo, "#000000"), linewidth=2)
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel("Episode Reward Mean", fontsize=12)
    plt.title("Multi-Agent Reinforcement Learning: Reward Convergence", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_reward_convergence.png"), dpi=150)
    plt.close()

    # Plot 2: Mean QBER
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        iterations = [r["iteration"] for r in algo_results[algo]]
        qbers = [r["mean_qber"] for r in algo_results[algo]]
        plt.plot(iterations, qbers, marker='s', label=algo, color=colors.get(algo, "#000000"), linewidth=2)
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel("Mean QBER", fontsize=12)
    plt.title("Quantum Bit Error Rate: Defense Effectiveness", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_mean_qber.png"), dpi=150)
    plt.close()

    # Plot 3: Mean SKR
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        iterations = [r["iteration"] for r in algo_results[algo]]
        skrs = [r["mean_skr"] for r in algo_results[algo]]
        plt.plot(iterations, skrs, marker='^', label=algo, color=colors.get(algo, "#000000"), linewidth=2)
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel("Mean Secret Key Rate (SKR)", fontsize=12)
    plt.title("Secret Key Rate: Network Throughput", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_mean_skr.png"), dpi=150)
    plt.close()

    # Plot 4: Detection F1 Score
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        iterations = [r["iteration"] for r in algo_results[algo]]
        f1_scores = [r["detection_f1_score"] for r in algo_results[algo]]
        plt.plot(iterations, f1_scores, marker='D', label=algo, color=colors.get(algo, "#000000"), linewidth=2)
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel("Detection F1 Score", fontsize=12)
    plt.title("Attack Detection Performance: F1 Score", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_detection_f1.png"), dpi=150)
    plt.close()

    # Plot 5: Detection AUC-ROC
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        iterations = [r["iteration"] for r in algo_results[algo]]
        auc_scores = [r["detection_auc_roc"] for r in algo_results[algo]]
        plt.plot(iterations, auc_scores, marker='v', label=algo, color=colors.get(algo, "#000000"), linewidth=2)
    plt.xlabel("Training Iteration", fontsize=12)
    plt.ylabel("Detection AUC-ROC", fontsize=12)
    plt.title("Attack Detection Performance: AUC-ROC", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_detection_auc.png"), dpi=150)
    plt.close()

    # Plot 6: Final Iteration Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    final_rewards = {algo: algo_results[algo][-1]["episode_reward_mean"] for algo in algorithms}
    axes[0, 0].bar(algorithms, final_rewards.values(), color=[colors.get(a, "#000000") for a in algorithms])
    axes[0, 0].set_ylabel("Episode Reward Mean", fontsize=11)
    axes[0, 0].set_title("Final Reward Comparison", fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    final_qbers = {algo: algo_results[algo][-1]["mean_qber"] for algo in algorithms}
    axes[0, 1].bar(algorithms, final_qbers.values(), color=[colors.get(a, "#000000") for a in algorithms])
    axes[0, 1].set_ylabel("Mean QBER", fontsize=11)
    axes[0, 1].set_title("Final QBER Comparison", fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    final_skrs = {algo: algo_results[algo][-1]["mean_skr"] for algo in algorithms}
    axes[1, 0].bar(algorithms, final_skrs.values(), color=[colors.get(a, "#000000") for a in algorithms])
    axes[1, 0].set_ylabel("Mean SKR", fontsize=11)
    axes[1, 0].set_title("Final SKR Comparison", fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    final_f1s = {algo: algo_results[algo][-1]["detection_f1_score"] for algo in algorithms}
    axes[1, 1].bar(algorithms, final_f1s.values(), color=[colors.get(a, "#000000") for a in algorithms])
    axes[1, 1].set_ylabel("Detection F1 Score", fontsize=11)
    axes[1, 1].set_title("Final F1 Score Comparison", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_final_comparison.png"), dpi=150)
    plt.close()

    print(f"\n[INFO] Plots saved to {output_dir}/")


def benchmark_all_algorithms(env_config: Dict[str, Any], num_iterations: int = 10):
    """
    Run comprehensive benchmark across all five MARL algorithms.
    """
    all_algos = ["IPPO", "MAPPO", "MADDPG", "QMIX", "VDN"]
    all_results = []

    print("\n" + "="*80)
    print("BENCHMARKING ALL FIVE MARL ALGORITHMS")
    print("="*80)

    for algo in all_algos:
        try:
            run_single_training(algo, env_config, num_iterations=num_iterations, num_rollout_workers=1, results_list=all_results)
        except Exception as exc:
            print(f"\n[ERROR] {algo} failed with exception: {exc}")
            import traceback
            traceback.print_exc()

    return all_results


# ============================================================
# 9) SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVERSARIAL MARL FOR ADAPTIVE DETECTION AND DEFENSE")
    print("AGAINST EAVESDROPPING IN DISTRIBUTED QKD NETWORKS")
    print("PhD Dissertation: Quantum Communication & Multi-Agent RL")
    print("="*80)

    # Environment configuration
    env_config = {
        "num_agents": 5,
        "max_steps": 20,
        "qber_base": 0.08,
        "qber_std": 0.02,
        "attack_probability": 0.30,
        "attack_intensity": 1.0,
        "attack_severity": 0.18,
        "seed": 42,
    }

    num_iterations = 10
    num_rollout_workers = 1

    # Run default algorithm (MAPPO)
    print(f"\n>>> Running default algorithm: MAPPO")
    print(f"    Environment: {env_config['num_agents']} agents, {env_config['max_steps']} steps per episode")
    print(f"    Attack probability: {env_config['attack_probability']:.0%}")
    print(f"    Training iterations: {num_iterations}")

    default_history = run_single_training(
        "MAPPO",
        env_config,
        num_iterations=num_iterations,
        num_rollout_workers=num_rollout_workers
    )

    # Run comprehensive benchmark across all algorithms
    print("\n>>> Running comprehensive benchmark across all algorithms...")
    benchmark_results = benchmark_all_algorithms(env_config, num_iterations=num_iterations)

    # Export results to CSV
    print("\n>>> Exporting results to CSV...")
    export_results_to_csv(benchmark_results, filename="qkd_marl_benchmark_results.csv")

    # Generate comparison plots
    print("\n>>> Generating comparison plots...")
    plot_benchmark_results(benchmark_results, output_dir="./qkd_marl_plots")

    # Print final summary
    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY")
    print("="*80)

    algo_summary = defaultdict(lambda: {
        "final_reward": 0.0,
        "final_qber": 0.0,
        "final_skr": 0.0,
        "final_f1": 0.0,
        "final_auc": 0.0,
    })

    for result in benchmark_results:
        algo = result["algorithm"]
        if result["iteration"] == num_iterations:
            algo_summary[algo]["final_reward"] = result["episode_reward_mean"]
            algo_summary[algo]["final_qber"] = result["mean_qber"]
            algo_summary[algo]["final_skr"] = result["mean_skr"]
            algo_summary[algo]["final_f1"] = result["detection_f1_score"]
            algo_summary[algo]["final_auc"] = result["detection_auc_roc"]

    print(f"\n{'Algorithm':<10} | {'Reward':>10} | {'QBER':>8} | {'SKR':>8} | {'F1':>8} | {'AUC':>8}")
    print("-" * 80)
    for algo in sorted(algo_summary.keys()):
        summary = algo_summary[algo]
        print(
            f"{algo:<10} | {summary['final_reward']:10.4f} | "
            f"{summary['final_qber']:8.4f} | {summary['final_skr']:8.4f} | "
            f"{summary['final_f1']:8.4f} | {summary['final_auc']:8.4f}"
        )

    print("\n" + "="*80)
    print("Benchmark complete. Results saved to:")
    print("  - CSV: qkd_marl_benchmark_results.csv")
    print("  - Plots: qkd_marl_plots/")
    print("="*80)
