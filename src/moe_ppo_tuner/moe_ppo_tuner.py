import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Iterable
from oppertune.core.values import Categorical, Integer, Real
from collections import deque
import random
import logging
from dataclasses import dataclass
import psutil
import time

torch.autograd.set_detect_anomaly(True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ("MoEPPOAlgorithm",)

@dataclass
class ExpertAction:
    discrete: Optional[Dict[str, int]] = None
    continuous: Optional[Dict[str, float]] = None

class DataProcessor:
    def __init__(self):
        self.latest_metrics = None

    def process_metrics(self, metrics: dict):
        self.latest_metrics = metrics
        state = np.zeros(64)
        state[0] = metrics.get('throughput', 0) / 10000
        state[1] = metrics.get('latency', 1000) / 1000
        return state

    def get_latest_state(self):
        if self.latest_metrics is None:
            return None
        return self.process_metrics(self.latest_metrics)

    def get_latest_metrics(self):
        return self.latest_metrics

class BottleneckIdentifier:
    def __init__(self, cpu_thres=0.10, mem_thres=0.10, io_thres=0.10, net_thres=0.5, window_size=3):
        self.cpu_thres = cpu_thres
        self.mem_thres = mem_thres
        self.io_thres = io_thres
        self.net_thres = net_thres
        self.max_disk_bw = 500 * 1024 * 1024
        self.max_net_bw = 125 * 1024 * 1024
        self.util_history = deque(maxlen=window_size)

    def get_system_util(self):
        cpu_util = psutil.cpu_percent(interval=1) / 100.0
        mem_util = psutil.virtual_memory().percent / 100.0
        disk_io1 = psutil.disk_io_counters()
        net1 = psutil.net_io_counters()
        time.sleep(1)
        disk_io2 = psutil.disk_io_counters()
        net2 = psutil.net_io_counters()
        io_bytes = ((disk_io2.read_bytes + disk_io2.write_bytes) -
                    (disk_io1.read_bytes + disk_io1.write_bytes))
        io_util = min(io_bytes / self.max_disk_bw, 1.0)
        net_bytes = ((net2.bytes_sent + net2.bytes_recv) -
                     (net1.bytes_sent + net1.bytes_recv))
        net_util = min(net_bytes / self.max_net_bw, 1.0)
        return cpu_util, mem_util, io_util, net_util

    def identify(self, state=None):
        util = self.get_system_util()
        self.util_history.append(util)
        if len(self.util_history) < self.util_history.maxlen:
            cpu_util, mem_util, io_util, net_util = util
        else:
            cpu_util = sum(u[0] for u in self.util_history) / len(self.util_history)
            mem_util = sum(u[1] for u in self.util_history) / len(self.util_history)
            io_util = sum(u[2] for u in self.util_history) / len(self.util_history)
            net_util = sum(u[3] for u in self.util_history) / len(self.util_history)
        logger.info(f"[Bottleneck] CPU: {cpu_util:.2f}, MEM: {mem_util:.2f}, IO: {io_util:.2f}, NET: {net_util:.2f}")
        bottleneck_vector = np.zeros(4)
        if cpu_util > self.cpu_thres:
            bottleneck_vector[0] = 1.0
        if mem_util > self.mem_thres:
            bottleneck_vector[1] = 1.0
        if io_util > self.io_thres:
            bottleneck_vector[2] = 1.0
        if net_util > self.net_thres:
            bottleneck_vector[3] = 1.0
        return bottleneck_vector

class ParameterAwareExpert(nn.Module):
    def __init__(self, input_dim: int, param_config: dict, bottleneck_subspace: List[str]):
        super().__init__()
        for param in bottleneck_subspace:
            if param not in param_config:
                raise KeyError(f"参数 {param} 不在 param_config 中")
        self.param_config = {k: v for k, v in param_config.items() if k in bottleneck_subspace}
        self.feature_dim = 64
        self.param_map = {k.replace('.', '_'): k for k in self.param_config.keys()}

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.feature_dim),
            nn.Tanh()
        )
        self.discrete_actor = nn.ModuleDict({
            k.replace('.', '_'): nn.Linear(self.feature_dim, len(config['options']))
            for k, config in self.param_config.items() if config['type'] in ['discrete', 'categorical']
        })
        self.continuous_actor = nn.ModuleDict({
            k.replace('.', '_'): nn.Linear(self.feature_dim, 2)
            for k, config in self.param_config.items() if config['type'] == 'continuous'
        })
        self.critic = nn.Linear(self.feature_dim, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

    def forward(self, state: torch.Tensor, sample: bool = True) -> Tuple[ExpertAction, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        discrete = {}
        continuous = {}
        log_probs = []
    
        batch_size = features.shape[0]
        for param in self.discrete_actor:
            logits = self.discrete_actor[param](features)  # [B, n_options]
            dist = torch.distributions.Categorical(logits=logits)
            if sample:
                action = dist.sample()  # [B]
                log_prob = dist.log_prob(action)  # [B]
            else:
                action = dist.probs.argmax(dim=-1)  # [B]
                log_prob = None
            orig_param = self.param_map[param]
    
            discrete[orig_param] = [self.param_config[orig_param]['options'][a.item()] for a in action] \
                if batch_size > 1 else self.param_config[orig_param]['options'][action.item()]
            if log_prob is not None:
                log_probs.append(log_prob)
    
        for param in self.continuous_actor:
            mu, logvar = self.continuous_actor[param](features).chunk(2, dim=-1)  # [B,1]
            std = torch.exp(0.5 * logvar)
            dist = torch.distributions.Normal(mu, std)
            if sample:
                action = dist.rsample()
                log_prob = dist.log_prob(action).squeeze(-1)
            else:
                action = mu
                log_prob = None
            orig_param = self.param_map[param]
            min_val = self.param_config[orig_param]['min']
            max_val = self.param_config[orig_param]['max']
            raw_action = min_val + (torch.tanh(action) + 1) * (max_val - min_val) / 2
            continuous[orig_param] = raw_action.squeeze(-1).tolist() if batch_size > 1 else raw_action.squeeze(-1).item()
            if log_prob is not None:
                log_probs.append(log_prob)
    
        if log_probs:
            combined_log_prob = sum(log_probs) if len(log_probs) == 1 else torch.stack(log_probs).sum(dim=0)
        else:
            combined_log_prob = torch.zeros(batch_size, device=state.device)
        value = self.critic(features)
        return ExpertAction(discrete=discrete, continuous=continuous), value, combined_log_prob


    def update(self, states, actions, rewards, next_states, dones, old_log_probs, clip_eps=0.2, gamma=0.99):
        features = self.feature_extractor(states)
        values = self.critic(features).squeeze(-1)
        next_values = self.critic(self.feature_extractor(next_states)).squeeze(-1)
        advantages = rewards + gamma * next_values * (1 - dones) - values
    
        for name, t in [('rewards', rewards), ('advantages', advantages), ('old_log_probs', old_log_probs)]:
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f'update: {name} has nan/inf:', t)
        advantages = torch.clamp(advantages, -1e6, 1e6)
        old_log_probs = torch.clamp(old_log_probs, -20, 20)
    
        _, _, new_log_probs = self(states, sample=False)
        if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
            print('update: new_log_probs nan/inf:', new_log_probs)
        new_log_probs = torch.clamp(new_log_probs, -20, 20)
    
        ratio = torch.exp(new_log_probs - old_log_probs)
        # clip ratio
        ratio = torch.clamp(ratio, 1e-6, 1e6)
    
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        value_loss = (values - (rewards + gamma * next_values * (1 - dones))).pow(2).mean()
        loss = actor_loss + 0.5 * value_loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("update: loss is nan/inf!", loss)
            return
    
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()


class GatedMoEPPO(nn.Module):
    def __init__(self, config: dict, state_dim: int, bottleneck_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.bottleneck_dim = bottleneck_dim
        self.experts = nn.ModuleList([
            ParameterAwareExpert(
                input_dim=state_dim,
                param_config=config['param_config'],
                bottleneck_subspace=subspace
            ) for subspace in config['subspaces']
        ])
        self.gate_actor = nn.Sequential(
            nn.Linear(state_dim + bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.experts)),
        )
        self.gate_critic = nn.Sequential(
            nn.Linear(state_dim + bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.gate_optimizer = torch.optim.AdamW(
            list(self.gate_actor.parameters()) + list(self.gate_critic.parameters()), lr=1e-4
        )
        self.memory = deque(maxlen=20000)
        self.batch_size = 4
        self.update_frequency = 5
        self.step = 0

    def forward(self, state: torch.Tensor, bottleneck_vector: torch.Tensor, sample: bool = True):
        gate_input = torch.cat([state, bottleneck_vector], dim=-1)
        gate_logits = self.gate_actor(gate_input)
        dist = torch.distributions.Categorical(logits=gate_logits)
        if sample:
            expert_idx = dist.sample()
            gate_log_prob = dist.log_prob(expert_idx)
        else:
            expert_idx = dist.probs.argmax(dim=-1)
            gate_log_prob = None

        expert = self.experts[expert_idx.item()]
        action, value, expert_log_prob = expert(state, sample=sample)
        gate_value = self.gate_critic(gate_input)
        return action, gate_value, gate_log_prob, expert_idx, expert_log_prob

    def store_transition(self, state, bottleneck, action_dict, reward, next_state, done, gate_log_prob, expert_idx, expert_log_prob):
        self.memory.append({
            'state': state.clone().detach(),
            'bottleneck': bottleneck.clone().detach(),
            'action': action_dict, 
            'reward': float(reward),
            'next_state': next_state.clone().detach(),
            'done': float(done),
            'gate_log_prob': gate_log_prob.clone().detach() if isinstance(gate_log_prob, torch.Tensor) else torch.tensor(0.0),
            'expert_idx': int(expert_idx),
            'expert_log_prob': expert_log_prob.clone().detach() if isinstance(expert_log_prob, torch.Tensor) else torch.tensor(0.0),
        })
        self.step += 1

    def update(self):
        if self.step < self.batch_size or self.step % self.update_frequency != 0:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([m['state'] for m in batch], dim=0)
        next_states = torch.cat([m['next_state'] for m in batch], dim=0)
        rewards = torch.tensor([m['reward'] for m in batch], dtype=torch.float32)
        dones = torch.tensor([m['done'] for m in batch], dtype=torch.float32)
        expert_idxs = [m['expert_idx'] for m in batch]
        expert_log_probs = torch.stack([m['expert_log_prob'] for m in batch])
        actions = [m['action'] for m in batch]
        for i, expert in enumerate(self.experts):
            idxs = [j for j, eid in enumerate(expert_idxs) if eid == i]
            if len(idxs) == 0: continue
            expert_states = states[idxs]
            expert_next_states = next_states[idxs]
            expert_rewards = rewards[idxs]
            expert_dones = dones[idxs]
            expert_actions = [actions[j] for j in idxs]
            expert_logprob = expert_log_probs[idxs]
            expert.update(
                expert_states,
                expert_actions,
                expert_rewards,
                expert_next_states,
                expert_dones,
                expert_logprob
            )
        print("Update done!")

class MoEPPOAlgorithm:
    def __init__(self, parameters: Iterable[Dict[str, Union[Integer, Real, Categorical, str]]]):
        self.parameters = parameters
        self.state_dim = 64
        self.param_config, self.subspaces = self._build_config()
        self.config = {
            'param_config': self.param_config,
            'subspaces': self.subspaces,
            'state_dim': self.state_dim
        }
        self.data_processor = DataProcessor()
        self.metric_mapping = {'throughput': 0, 'latency': 1}
        self.bottleneck_identifier = BottleneckIdentifier(cpu_thres=0.8, mem_thres=0.8, io_thres=0.8, net_thres=0.8)
        self.model = GatedMoEPPO(self.config, self.state_dim, bottleneck_dim=4)
        self.model.memory.clear()
        self.prev_metrics = None
        self._iteration = 0
        self.last_prediction = {p['param'].name: p['param'].val for p in self.parameters}

    def _build_config(self):
        param_config = {}
        subspaces = [[], [], [], []]
        expert_map = {'CPU': 0, 'MEM': 1, 'IO': 2, 'NET': 3}
        for item in self.parameters:
            param = item['param']
            expert_type = item.get('expert_type', 'CPU')
            name = param.name
            if isinstance(param, Integer):
                options = list(range(param.min, param.max + 1))
                param_config[name] = {
                    'type': 'discrete',
                    'options': options,
                    'default': param.val,
                    'min': param.min,
                    'max': param.max
                }
            elif isinstance(param, Real):
                param_config[name] = {
                    'type': 'continuous',
                    'min': param.min,
                    'max': param.max,
                    'default': param.val
                }
            elif isinstance(param, Categorical):
                param_config[name] = {
                    'type': 'categorical',
                    'options': param.categories,
                    'default': param.val
                }
            else:
                logger.warning(f"Unknown parameter type {type(param)}, skipping {name}")
                continue
            if expert_type in expert_map:
                subspaces[expert_map[expert_type]].append(name)
            else:
                logger.warning(f"Unknown expert_type '{expert_type}', assigning {name} to cpu_expert")
                subspaces[0].append(name)
        for i, subspace in enumerate(subspaces):
            if not subspace:
                logger.warning(f"Subspace {i} is empty, assigning default parameter")
                if param_config:
                    subspaces[i].append(list(param_config.keys())[0])
                else:
                    logger.error("No parameters available to assign to subspace {i}")
                    raise ValueError(f"Cannot initialize subspace {i} with no parameters")
        logger.info(f"Subspace configuration: {subspaces}")
        return param_config, subspaces

    def predict(self) -> Dict[str, Union[int, float, str]]:
        metrics = self.data_processor.get_latest_metrics()
        if metrics is None:
            return {p['param'].name: p['param'].val for p in self.parameters}
        system_state = np.zeros(self.state_dim)
        system_state[0] = metrics.get('throughput', 0) / 10000
        system_state[1] = metrics.get('latency', 1000) / 1000
        state = torch.FloatTensor(system_state).unsqueeze(0)
        bottleneck_vector = self.bottleneck_identifier.identify(system_state)
        bottleneck_tensor = torch.FloatTensor(bottleneck_vector).unsqueeze(0)
        action, gate_value, gate_log_prob, expert_idx, expert_log_prob = self.model(state, bottleneck_tensor, sample=True)
        prediction = self._action_to_dict(action)
        full_prediction = self.last_prediction.copy()
        full_prediction.update(prediction)
        self.last_prediction = full_prediction
        reward = self.calculate_reward(metrics)
        self.model.store_transition(
            state, bottleneck_tensor, prediction, reward, state, False, gate_log_prob, expert_idx, expert_log_prob
        )
        self.model.update()
        self.prev_metrics = metrics.copy()
        self._iteration += 1
        return full_prediction

    def set_reward(self, reward: float, metrics: Dict[str, float]) -> None:
        if reward is None:
            raise ValueError("reward cannot be None for MoEPPOAlgorithm")
        self.data_processor.process_metrics(metrics)
        self._iteration += 1

    def calculate_reward(self, metrics):
        throughput = metrics.get("throughput", 0)
        latency = metrics.get("latency", 1000)
        return np.clip((throughput / (latency + 1)) / 100, -1.0, 1.0)

    def _action_to_dict(self, action: ExpertAction) -> Dict[str, Union[int, float]]:
        result = {}
        if action.discrete:
            result.update(action.discrete)
        if action.continuous:
            result.update(action.continuous)
        return result

    @property
    def iteration(self) -> int:
        return self._iteration

