"""Utility functions for FoodBridge AI project."""

import os
import random
import logging
import numpy as np
import json
from typing import Any, Dict
from pathlib import Path

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger with given name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except:
        pass

def create_directories(paths: list):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent

logger = setup_logger(__name__)
