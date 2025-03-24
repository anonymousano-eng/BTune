import subprocess
import logging
import numpy as np
from typing import Dict
from .config_loader import ConfigLoader, ParameterConfig

class ParameterManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def apply_actions(self, actions: Dict):
        for expert_name, params in actions.items():
            for param_name, value in params.items():
                config = self._get_param_config(expert_name, param_name)
                if config is None:
                    self.logger.warning(f"Parameter {param_name} not found in {expert_name}")
                    continue
                try:
                    self._set_parameter(config, value)
                except Exception as e:
                    self.logger.error(f"Failed to set {param_name}: {str(e)}")
    
    def _get_param_config(self, expert_name, param_name):
        for expert in self.config.experts:
            if expert.name == expert_name:
                return expert.parameters.get(param_name)
        self.logger.warning(f"Expert {expert_name} not found in configuration")
        return None
    
    def _set_parameter(self, config: ParameterConfig, value: float):
        try:
            if config.param_type == 'continuous':
                scaled_value = self._scale_continuous_value(config, value)
                cmd = f"sysctl -w {config.name}={scaled_value}"
            else:
                idx = int(value * len(config.options))
                idx = max(0, min(idx, len(config.options)-1))
                selected = config.options[idx]
                cmd = f"sysctl -w {config.name}={selected}"
                
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"Set {config.name} to {value}: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set {config.name}: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Unexpected error setting {config.name}: {str(e)}")
    
    def _scale_continuous_value(self, config: ParameterConfig, value: float):
        value = np.clip(value, -5, 5)
        sigmoid = 1 / (1 + np.exp(-value))
        scaled = config.min_val + sigmoid * (config.max_val - config.min_val)
        return np.clip(scaled, config.min_val, config.max_val)