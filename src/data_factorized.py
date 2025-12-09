"""
Factorized Input Data Generation

KEY CHANGE: Inputs are now FACTORIZED (concatenated one-hots for each factor)
rather than one-hot over all combinations.

This provides RECOMBINATION EXPOSURE:
- "red circle" and "red square" share the first 10 input dimensions
- The network can SEE that these inputs share structure

This tests APH Section 9.2 more fairly:
- Factorization pressure (bottleneck) ✓
- Recombination exposure (factorized input) ✓
- Multi-task learning (optional) ✓

Scientific question: Does bottleneck pressure + recombination exposure
yield factorized representations and compositional generalization?

Reference: This design is similar to Lake & Baroni (2018) SCAN benchmark,
where inputs have compositional structure that the network could exploit.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass 
class FactorizedDataConfig:
    """Configuration for factorized input dataset."""
    n_factor1: int = 10
    n_factor2: int = 10
    holdout_fraction: float = 0.25
    holdout_strategy: str = "compositional"
    seed: int = 42


class FactorizedInputDataset:
    """
    Factorial dataset with FACTORIZED input representation.
    
    Key difference from FactorialDataset:
    - Input is concatenated one-hots: [factor1_onehot, factor2_onehot]
    - Dimension: n_factor1 + n_factor2 (not n_factor1 * n_factor2)
    
    This means:
    - "red circle" = [1,0,0,...,0, 1,0,0,...,0] (red one-hot + circle one-hot)
    - "red square" = [1,0,0,...,0, 0,1,0,...,0] (red one-hot + square one-hot)
    
    The network can SEE that these share the "red" component!
    
    Output is still classification over all combinations (tests composition).
    """
    
    def __init__(self, config: FactorizedDataConfig):
        self.config = config
        self.n_factor1 = config.n_factor1
        self.n_factor2 = config.n_factor2
        self.n_combinations = config.n_factor1 * config.n_factor2
        self.input_dim = config.n_factor1 + config.n_factor2  # KEY CHANGE
        
        np.random.seed(config.seed)
        
        # Generate all combinations
        self.all_combinations = [
            (i, j)
            for i in range(self.n_factor1)
            for j in range(self.n_factor2)
        ]
        
        # Index mappings
        self.combination_to_idx = {c: i for i, c in enumerate(self.all_combinations)}
        self.idx_to_combination = {i: c for c, i in self.combination_to_idx.items()}
        
        # Create split
        self._create_split()
    
    def _create_split(self) -> None:
        """Create compositional holdout split."""
        n_holdout = int(self.n_combinations * self.config.holdout_fraction)
        
        if self.config.holdout_strategy == "compositional":
            factor1_counts = defaultdict(int)
            factor2_counts = defaultdict(int)
            for f1, f2 in self.all_combinations:
                factor1_counts[f1] += 1
                factor2_counts[f2] += 1
            
            candidates = [
                (f1, f2) for f1, f2 in self.all_combinations
                if factor1_counts[f1] > 1 and factor2_counts[f2] > 1
            ]
            np.random.shuffle(candidates)
            
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
        else:
            shuffled = self.all_combinations.copy()
            np.random.shuffle(shuffled)
            self.test_combinations = shuffled[:n_holdout]
            self.train_combinations = shuffled[n_holdout:]
    
    def get_input(self, combination: Tuple[int, int]) -> torch.Tensor:
        """
        Get FACTORIZED input representation.
        
        Returns concatenation of one-hot vectors for each factor.
        Dimension: n_factor1 + n_factor2
        """
        f1, f2 = combination
        
        # One-hot for factor 1
        f1_onehot = torch.zeros(self.n_factor1)
        f1_onehot[f1] = 1.0
        
        # One-hot for factor 2
        f2_onehot = torch.zeros(self.n_factor2)
        f2_onehot[f2] = 1.0
        
        # Concatenate
        return torch.cat([f1_onehot, f2_onehot])
    
    def get_target(self, combination: Tuple[int, int]) -> torch.Tensor:
        """Target is combination index (for composition task)."""
        return torch.tensor(self.combination_to_idx[combination])
    
    def get_factor_targets(
        self, combination: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get individual factor indices."""
        f1, f2 = combination
        return torch.tensor(f1), torch.tensor(f2)
    
    def get_batch(
        self,
        split: str = "train",
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of data."""
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
        return (
            f"Factorized Input Dataset: {self.n_factor1} × {self.n_factor2}\n"
            f"Input dim: {self.input_dim} (factorized) vs {self.n_combinations} (holistic)\n"
            f"Train: {len(self.train_combinations)} | Test: {len(self.test_combinations)}"
        )