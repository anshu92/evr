"""Configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .schema import Config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file or return default config.
    
    Args:
        config_path: Path to YAML configuration file. If None, returns default config.
        
    Returns:
        Config object with loaded settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If config validation fails.
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    config = Config.from_dict(data)
    config.validate()
    
    return config


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        config_path: Path to save configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    return Path(__file__).parent / "defaults.yaml"


def load_default_config() -> Config:
    """Load default configuration."""
    default_path = get_default_config_path()
    if default_path.exists():
        return load_config(default_path)
    return Config()
