"""Configuration module for FoodBridge AI project."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Load default config
CONFIG = load_config()
