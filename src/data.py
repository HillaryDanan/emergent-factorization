"""
Data generation for factorial structure experiments.

Key design principle: Input representation is intentionally NOT factorized.
The network must DISCOVER factorization from task/capacity pressure.

This is critical for testing the hypothesis that factorization EMERGES
rather than being handed to the system.

Theoretical basis:
- Compositional generalization: Lake & Baroni (2018), ICML
- Disentanglement requires inductive bias: Locatello et al. (2019), ICML
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for factorial dataset."""
    n_factor1: int = 10  # e.g., number of colors
    n_factor2: int = 10  # e.g., number of shapes
    holdout_fraction: float = 0.25
    holdout_strategy: str = "compositional"  # or "random"
    seed: int = 42


class FactorialDataset:
    """
    Generates factorial data (e.g., color × shape combinations).
    
    IMPORTANT DESIGN CHOICE:
    Inputs are one-hot over ALL combinations, NOT factorized.
    This means input for "red circle" is a single bit in a 100-dim vector,
    NOT [1,0,0,...] concatenated with [1,0,0,...].
    
    Why? We want to test if the network DISCOVERS factorization.
    If we give factorized inputs, we're not testing emergence.
    
    Holdout strategies:
    - "compositional": Hold out combinations where BOTH factors seen separately
      (tests compositional generalization specifically)
    - "random": Random holdout (control condition)
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.n_factor1 = config.n_factor1
        self.n_factor2 = config.n_factor2
        self.n_combinations = config.n_factor1 * config.n_factor2
        
        np.random.seed(config.seed)
        
        # Generate all combinations as (factor1_idx, factor2_idx) pairs
        self.all_combinations = [
            (i, j) 
            for i in range(self.n_factor1) 
            for j in range(self.n_factor2)
        ]
        
        # Create index mapping
        self.combination_to_idx = {c: i for i, c in enumerate(self.all_combinations)}
        self.idx_to_combination = {i: c for c, i in self.combination_to_idx.items()}
        
        # Create train/test split
        self._create_split()
    
    def _create_split(self) -> None:
        """
        Create train/test split based on strategy.
        
        Compositional holdout ensures:
        1. Each factor value appears in training
        2. Held-out combinations are NOVEL compositions of known factors
        
        This is the key test: can the system generalize to new combinations
        of familiar components?
        """
        n_holdout = int(self.n_combinations * self.config.holdout_fraction)
        
        if self.config.holdout_strategy == "compositional":
            # Track factor occurrences
            factor1_counts = defaultdict(int)
            factor2_counts = defaultdict(int)
            for f1, f2 in self.all_combinations:
                factor1_counts[f1] += 1
                factor2_counts[f2] += 1
            
            # Find candidates we can hold out (both factors appear elsewhere)
            candidates = [
                (f1, f2) for f1, f2 in self.all_combinations
                if factor1_counts[f1] > 1 and factor2_counts[f2] > 1
            ]
            np.random.shuffle(candidates)
            
            # Select holdout while ensuring all factors remain in training
            self.test_combinations = []
            temp_f1 = factor1_counts.copy()
            temp_f2 = factor2_counts.copy()
            
            for f1, f2 in candidates:
                if len(self.test_combinations) >= n_holdout:
                    break
                if temp_f1[f1] > 1 and temp_f2[f2] > 1:
                    self.test_combinations.append((f1, f2))
                    temp_f1[f1] -= 1
                    temp_f2[f2] -= 1
            
            self.train_combinations = [
                c for c in self.all_combinations 
                if c not in self.test_combinations
            ]
            
        elif self.config.holdout_strategy == "random":
            shuffled = self.all_combinations.copy()
            np.random.shuffle(shuffled)
            self.test_combinations = shuffled[:n_holdout]
            self.train_combinations = shuffled[n_holdout:]
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.holdout_strategy}")
    
    def get_input(self, combination: Tuple[int, int]) -> torch.Tensor:
        """
        Get one-hot input for a combination.
        
        NOTE: This is one-hot over ALL combinations (e.g., 100-dim for 10x10).
        The network sees NO structural hint about factorization.
        """
        idx = self.combination_to_idx[combination]
        one_hot = torch.zeros(self.n_combinations)
        one_hot[idx] = 1.0
        return one_hot
    
    def get_target(self, combination: Tuple[int, int]) -> torch.Tensor:
        """Target is the combination index."""
        return torch.tensor(self.combination_to_idx[combination])
    
    def get_factor_targets(
        self, combination: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get individual factor indices (for multi-task and analysis)."""
        f1, f2 = combination
        return torch.tensor(f1), torch.tensor(f2)
    
    def get_batch(
        self, 
        split: str = "train", 
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.
        
        Returns:
            inputs: [batch, n_combinations] one-hot
            targets: [batch] combination indices
            factor1_targets: [batch] factor 1 indices (for multi-task)
            factor2_targets: [batch] factor 2 indices (for multi-task)
        """
        combinations = (
            self.train_combinations if split == "train" 
            else self.test_combinations
        )
        
        if batch_size is None or batch_size >= len(combinations):
            batch = combinations
        else:
            indices = np.random.choice(len(combinations), batch_size, replace=False)
            batch = [combinations[i] for i in indices]
        
        inputs = torch.stack([self.get_input(c) for c in batch])
        targets = torch.stack([self.get_target(c) for c in batch])
        f1 = torch.stack([self.get_factor_targets(c)[0] for c in batch])
        f2 = torch.stack([self.get_factor_targets(c)[1] for c in batch])
        
        return inputs, targets, f1, f2
    
    def summary(self) -> str:
        """Dataset summary string."""
        return (
            f"Factorial Dataset: {self.n_factor1} × {self.n_factor2} "
            f"= {self.n_combinations} combinations\n"
            f"Train: {len(self.train_combinations)} | "
            f"Test: {len(self.test_combinations)}\n"
            f"Strategy: {self.config.holdout_strategy}"
        )
    
    def get_theoretical_dimensions(self) -> Dict[str, int]:
        """
        Return theoretical dimension requirements.
        
        Key insight from APH framework:
        - Factorized representation needs: n_factor1 + n_factor2 dimensions
        - Holistic memorization needs: n_factor1 × n_factor2 dimensions
        """
        return {
            "sum_factors": self.n_factor1 + self.n_factor2,
            "product_factors": self.n_combinations,
            "factorization_ratio": self.n_combinations / (self.n_factor1 + self.n_factor2)
        }