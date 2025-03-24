import yaml
from dataclasses import dataclass
from typing import List, Dict, Union

@dataclass
class ParameterConfig:
    name: str
    param_type: str  # continuous/discrete
    min_val: float = None
    max_val: float = None
    options: List[str] = None
    step: float = None

@dataclass
class ExpertConfig:
    name: str
    parameters: Dict[str, ParameterConfig]

class ConfigLoader:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.raw_config = yaml.safe_load(f)
            
        self.experts = []
        self._parse_config()
        
    def _parse_config(self):
        for expert_name, params in self.raw_config['experts'].items():
            param_configs = {}
            for param_name, config in params['parameters'].items():
                param_configs[param_name] = ParameterConfig(
                    name=param_name,
                    param_type=config['type'],
                    min_val=config.get('min'),
                    max_val=config.get('max'),
                    options=config.get('options'),
                    step=config.get('step')
                )
            self.experts.append(ExpertConfig(
                name=expert_name,
                parameters=param_configs
            ))
            
    def get_expert_configs(self):
        return self.experts

    def get_parameter_space(self):
        action_dims = {'discrete': 0, 'continuous': 0}
        for expert in self.experts:
            for param in expert.parameters.values():
                if param.param_type == 'discrete':
                    action_dims['discrete'] += len(param.options)
                else:
                    action_dims['continuous'] += 1
        return action_dims