"""
Utility functions for experiments.

Keep it simple, keep it robust.
"""

import torch
import numpy as np
import random
import os
from datetime import datetime
from typing import Dict, Any
import json


def set_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Reproducibility is essential for scientific validity.
    Reference: Pineau et al. (2021) "Improving Reproducibility in ML Research"
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Detect best available device for Mac."""
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
        return "mps"
    elif torch.cuda.is_available():
        print("Using CUDA")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)  # Fallback for any other types


def save_results(results: Dict, save_dir: str, prefix: str) -> str:
    """Save results to JSON with timestamp."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"{prefix}_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict:
    """Load results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)