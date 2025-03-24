import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExpertAction:
    discrete: Optional[Dict[str, int]] = None
    continuous: Optional[Dict[str, float]] = None

class ParameterAwareExpert(nn.Module):
    def __init__(self, input_dim: int, param_config: dict):
        super().__init__()
        self.param_config = param_config
        self.feature_dim = 64
        
        self.discrete_dims = sum(
            len(v['options']) if v['type'] == 'discrete' else 1
            for v in param_config.values()
        )
        self.continuous_dims = sum(
            1 for v in param_config.values() if v['type'] == 'continuous'
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.feature_dim),
            nn.Tanh()
        )
        
        self.discrete_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.discrete_dims)
        )
        
        self.continuous_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.continuous_dims * 2)
        )

    def forward(self, x: torch.Tensor) -> ExpertAction:
        features = self.feature_extractor(x)
        
        discrete_logits = self.discrete_head(features)
        
        continuous_params = self.continuous_head(features)
        mu, logvar = torch.chunk(continuous_params, 2, dim=-1)
        
        return ExpertAction(
            discrete=self._map_discrete_actions(discrete_logits),
            continuous=self._map_continuous_actions(mu, logvar)
        )

    def _map_discrete_actions(self, logits: torch.Tensor) -> Dict[str, int]:
        actions = {}
        ptr = 0
        for param_name, config in self.param_config.items():
            if config['type'] == 'discrete':
                n_options = len(config['options'])
                probs = F.softmax(logits[ptr:ptr+n_options], dim=-1)
                actions[param_name] = torch.argmax(probs).item()
                ptr += n_options
        return actions

    def _map_continuous_actions(self, mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, float]:
        actions = {}
        ptr = 0
        for param_name, config in self.param_config.items():
            if config['type'] == 'continuous':
                std = torch.exp(0.5 * logvar[ptr])
                action = torch.tanh(mu[ptr] + torch.randn_like(std) * std)
                actions[param_name] = action.item()
                ptr += 1
        return actions

class BottleneckAwareGate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
               lstm_features: torch.Tensor, 
               bottleneck_vector: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([lstm_features, bottleneck_vector], dim=-1)
        return self.gate_network(combined)

class GatedCollaborativeRL:
    def __init__(self, config, state_dim: int):
        self.config = config
        self.experts = nn.ModuleDict()
        
        for expert in config['expert_configs']:
            self.experts[expert.name] = ParameterAwareExpert(
                input_dim=state_dim,
                param_config={param.name: {
                    'type': param.param_type,
                    'min': param.min_val,
                    'max': param.max_val,
                    'options': param.options,
                    'step': param.step
                } for param in expert.parameters.values()}
            )
        
        self.gate = BottleneckAwareGate(
            input_dim=state_dim + 8,
            num_experts=len(config['expert_configs'])
        )
        
        self.optimizer = torch.optim.AdamW(
            list(self.experts.parameters()) + list(self.gate.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01

    def get_actions(self, 
                   system_state: np.ndarray,
                   lstm_features: np.ndarray,
                   bottleneck_vector: np.ndarray) -> Dict[str, ExpertAction]:
        state_tensor = torch.FloatTensor(system_state)
        lstm_tensor = torch.FloatTensor(lstm_features)
        bottleneck_tensor = torch.FloatTensor(bottleneck_vector)
        
        gate_weights = self.gate(lstm_tensor, bottleneck_tensor)
        
        selected_experts = torch.topk(gate_weights, k=2).indices
        actions = {}
        
        for idx in selected_experts:
            expert_name = list(self.experts.keys())[idx.item()]
            expert = self.experts[expert_name]
            actions[expert_name] = expert(state_tensor)
            
        return actions

    def update(self, batch_data: List[Dict]):
        states = torch.stack([torch.FloatTensor(d['state']) for d in batch_data])
        old_log_probs = torch.stack([d['log_prob'] for d in batch_data])
        advantages = torch.FloatTensor([d['advantage'] for d in batch_data]).detach()
        returns = torch.FloatTensor([d['return'] for d in batch_data]).detach()

        new_log_probs = []
        for state in states:
            gate_weights = self.gate(state.unsqueeze(0), torch.FloatTensor([0.0]*8))
            selected_experts = torch.topk(gate_weights, k=2).indices
            log_prob = 0.0
            for idx in selected_experts:
                expert_name = list(self.experts.keys())[idx.item()]
                expert = self.experts[expert_name]
                action = expert(state.unsqueeze(0))
                
                if action.discrete:
                    ptr = 0
                    for param, value in action.discrete.items():
                        config = {param.name: {
                            'type': param.param_type,
                            'min': param.min_val,
                            'max': param.max_val,
                            'options': param.options,
                            'step': param.step
                        } for param in self.config['expert_configs'][idx.item()].parameters.values()}[param]
                        n_options = len(config['options'])
                        logits = expert.discrete_head(expert.feature_extractor(state.unsqueeze(0)))
                        probs = F.softmax(logits[0][ptr:ptr+n_options], dim=-1)
                        log_prob += torch.log(probs[value])
                        ptr += n_options
                
                if action.continuous:
                    ptr = 0
                    mu, logvar = expert.continuous_head(expert.feature_extractor(state.unsqueeze(0))).chunk(2, dim=-1)
                    for param, value in action.continuous.items():
                        config = {param.name: {
                            'type': param.param_type,
                            'min': param.min_val,
                            'max': param.max_val,
                            'options': param.options,
                            'step': param.step
                        } for param in self.config['expert_configs'][idx.item()].parameters.values()}[param]
                        std = torch.exp(0.5 * logvar[0][ptr])
                        dist = torch.distributions.Normal(mu[0][ptr], std)
                        log_prob += dist.log_prob(torch.tensor(value))
                        ptr += 1
            new_log_probs.append(log_prob)
        new_log_probs = torch.stack(new_log_probs)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        entropy = torch.mean(-new_log_probs)

        total_loss = actor_loss + self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

    def save_checkpoint(self, path: str):
        torch.save({
            'experts': self.experts.state_dict(),
            'gate': self.gate.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.experts.load_state_dict(checkpoint['experts'])
        self.gate.load_state_dict(checkpoint['gate'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class HybridActionNormalizer:
    def __init__(self, config):
        self.config = config
        self._build_normalization_rules()
        
    def _build_normalization_rules(self):
        self.norm_rules = {}
        for expert in config['expert_configs']:
            for param_name, param_config in expert.parameters.items():
                if param_config.param_type == 'continuous':
                    self.norm_rules[param_name] = (
                        param_config.min_val, 
                        param_config.max_val
                    )
                else:
                    self.norm_rules[param_name] = param_config.options
                    
    def normalize(self, actions: Dict[str, ExpertAction]) -> Dict[str, ExpertAction]:
        normalized = {}
        for expert_name, action in actions.items():
            norm_action = ExpertAction()
            if action.discrete:
                norm_action.discrete = {}
                for param, value in action.discrete.items():
                    options = self.norm_rules[param]
                    norm_value = (value - 0) / (len(options)-1) * 2 - 1
                    norm_action.discrete[param] = norm_value
            if action.continuous:
                norm_action.continuous = {}
                for param, value in action.continuous.items():
                    min_val, max_val = self.norm_rules[param]
                    norm_value = (value - min_val) / (max_val - min_val) * 2 - 1
                    norm_action.continuous[param] = norm_value
            normalized[expert_name] = norm_action
        return normalized

    def denormalize(self, norm_actions: Dict[str, ExpertAction]) -> Dict[str, ExpertAction]:
        denorm = {}
        for expert_name, action in norm_actions.items():
            denorm_action = ExpertAction()
            if action.discrete:
                denorm_action.discrete = {}
                for param, value in action.discrete.items():
                    options = self.norm_rules[param]
                    idx = int((value + 1) * (len(options)-1) / 2)
                    denorm_action.discrete[param] = options[idx]
            if action.continuous:
                denorm_action.continuous = {}
                for param, value in action.continuous.items():
                    min_val, max_val = self.norm_rules[param]
                    denorm_value = min_val + (value + 1) * (max_val - min_val) / 2
                    denorm_action.continuous[param] = denorm_value
            denorm[expert_name] = denorm_action
        return denorm
    