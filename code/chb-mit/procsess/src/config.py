import yaml
from pathlib import Path

class Config:
    def __init__(self, data):
        self._data = data
        for key, val in data.items():
            if isinstance(val, dict):
                setattr(self, key, Config(val))
            else:
                setattr(self, key, val)

    def get(self, key, default=None):
        return self._data.get(key, default)

def load_config(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(data)
