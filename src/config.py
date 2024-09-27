# src/config.py
import yaml

def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
