import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _validate_config(self):
        required_sections = ['data', 'model', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

        if 'num_classes' not in self.config['data']:
            raise ValueError("num_classes must be defined in data section")

        if self.config['data']['num_classes'] == 1:
            self.config['model']['loss'] = 'binary_crossentropy'
            self.config['model']['activation'] = 'sigmoid'
        else:
            self.config['model']['loss'] = 'categorical_crossentropy'
            self.config['model']['activation'] = 'softmax'

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def update(self, key: str, value):
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

    def save(self, path: str = None):
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
