import os
from dataclasses import dataclass
from typing import Optional

import torch
import yaml


@dataclass
class Config:
    """
    Defines configuration fields, including all you'd normally
    store in config.yaml, plus dynamically determined fields
    like 'device' and any environment-based fields (e.g. the OpenAI API key).
    """
    dataset: str
    split: str
    model: str
    n_test: int
    batch_size: int
    do_perturb: bool
    n_perturb: int
    n_sample: int
    temperature: float
    max_tokens: int
    eigenvalues_threshold: float
    graph_mode: Optional[str]
    stop: Optional[str]
    prompt_type: str
    preload_responses: bool
    preload_accuracy: bool
    preload_uncertainty: bool
    preload_calibration: bool
    save_results: bool
    verbose: bool
    device: torch.device
    openai_api_key: str


def load_config(config_path) -> Config:
    """
    Loads configuration from config.yaml, applies device logic, 
    appends the OpenAI API key from the environment, 
    and returns a typed Config object.
    """
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_conf = yaml.safe_load(f)

    requested_device = raw_conf.pop("device", "auto")
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    raw_conf["device"] = device

    # Grabs the OpenAI API key from the environment
    raw_conf["openai_api_key"] = os.environ["OPENAI_API_KEY"]

    return Config(**raw_conf)


# Module-level config instance for import across the codebase
# config = load_config()