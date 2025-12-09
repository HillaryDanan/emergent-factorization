"""
EXPERIMENT: Asymmetry Boundary Probing

PURPOSE:
Extended experiments revealed that factor asymmetry catastrophically degrades
compositional generalization. This follow-up systematically probes the boundary.

FINDINGS TO EXPLAIN:
- [10, 10] (ratio 1.0): 80%
- [5, 20] (ratio 4.0): 30%
- [4, 25] (ratio 6.25): 5%
- [2, 50] (ratio 25.0): 0%

QUESTIONS:
1. What is the precise boundary ratio?
2. Is it asymmetry ratio or compression ratio that matters more?
3. Can increased training data compensate for asymmetry?
4. Can auxiliary losses enforce balanced representations?

EXPERIMENTAL DESIGN:
1. Fine-grained asymmetry sweep: ratios 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0
2. Matched controls: same compression ratio, different asymmetry
3. Data compensation: more training data for high-cardinality factors
4. Auxiliary loss: encourage balanced factor representations

Design principles:
- Systematic parameter sweep
- Matched controls to isolate variables
- Clear falsification criteria
- Parsimony: simplest experiments first
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import product
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seeds, get_device, save_results


# ==============================================================================
# DATA AND MODEL (Reuse flexible N-factor infrastructure)
# ==============================================================================

@dataclass
class AsymmetryConfig:
    """Configuration for asymmetry experiments."""
    factor_cardinalities: List[int]
    holdout_fraction: float = 0.25
    oversample_rare: bool = False  # Oversample rare factor values
    oversample_factor: float = 1.0  # How much to oversample
    seed: int = 42
    
    @property
    def n_factors(self) -> int:
        return len(self.factor_cardinalities)
    
    @property
    def n_combinations(self) -> int:
        return math.prod(self.factor_cardinalities)
    
    @property
    def input_dim(self) -> int:
        return sum(self.factor_cardinalities)
    
    @property
    def asymmetry_ratio(self) -> float:
        return max(self.factor_cardinalities) / min(self.factor_cardinalities)
    
    @property
    def compression_ratio(self) -> float:
        return self.n_combinations / self.input_dim


class AsymmetryDataset:
    """Dataset for asymmetry experiments with optional oversampling."""
    
    def __init__(self, config: AsymmetryConfig):
        self.config = config
        self.cardinalities = config.factor_cardinalities
        self.n_factors = config.n_factors
        self.n_combinations = config.n_combinations
        self.input_dim = config.input_dim
        
        np.random.seed(config.seed)
        
        self.all_combinations = list(product(*[range(c) for c in self.cardinalities]))
        self._create_split()
        
        if config.oversample_rare:
            self._create_oversampled_training()
    
    def _create_split(self):
        """Compositional holdout."""
        n_holdout = int(self.n_combinations * self.config.holdout_fraction)
        
        factor_counts = [defaultdict(int) for _ in range(self.n_factors)]
        for combo in self.all_combinations:
            for f_idx, f_val in enumerate(combo):
                factor_counts[f_idx][f_val] += 1
        
        candidates = [
            combo for combo in self.all_combinations
            if all(factor_counts[f_idx][f_val] > 1 
                   for f_idx, f_val in enumerate(combo))
        ]
        np.random.shuffle(candidates)
        
        self.test_combinations = []
        temp_counts = [fc.copy() for fc in factor_counts]
        
        for combo in candidates:
            if len(self.test_combinations) >= n_holdout:
                break
            if all(temp_counts[f_idx][f_val] > 1 
                   for f_idx, f_val in enumerate(combo)):
                self.test_combinations.append(combo)
                for f_idx, f_val in enumerate(combo):
                    temp_counts[f_idx][f_val] -= 1
        
        self.train_combinations = [
            c for c in self.all_combinations if c not in self.test_combinations
        ]
        
        # Default: training set is just the combinations
        self.training_samples = self.train_combinations.copy()
    
    def _create_oversampled_training(self):
        """Oversample combinations containing rare factor values."""
        # Find the highest cardinality factor
        max_card = max(self.cardinalities)
        max_factor_idx = self.cardinalities.index(max_card)
        
        # Calculate how many times each combination should appear
        # based on how rare its high-cardinality factor value is
        oversample_rate = self.config.oversample_factor
        
        self.training_samples = []
        for combo in self.train_combinations:
            # Base: include once
            self.training_samples.append(combo)
            
            # For high-cardinality factors, add extra copies
            # The rarer the value, the more copies
            high_card_value = combo[max_factor_idx]
            # Count how many training combos have this value
            count = sum(1 for c in self.train_combinations 
                       if c[max_factor_idx] == high_card_value)
            
            # If rare, oversample
            if count < 3:  # Very rare
                extra_copies = int(oversample_rate * 3)
            elif count < 5:  # Somewhat rare
                extra_copies = int(oversample_rate * 1)
            else:
                extra_copies = 0
            
            for _ in range(extra_copies):
                self.training_samples.append(combo)
    
    def get_input(self, combination: Tuple) -> torch.Tensor:
        """Factorized input."""
        parts = []
        for f_idx, f_val in enumerate(combination):
            one_hot = torch.zeros(self.cardinalities[f_idx])
            one_hot[f_val] = 1.0
            parts.append(one_hot)
        return torch.cat(parts)
    
    def get_factor_targets(self, combination: Tuple) -> List[torch.Tensor]:
        return [torch.tensor(f_val) for f_val in combination]
    
    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if split == "train":
            combinations = self.training_samples  # May be oversampled
        else:
            combinations = self.test_combinations
        
        inputs = torch.stack([self.get_input(c) for c in combinations])
        factor_targets = [
            torch.stack([self.get_factor_targets(c)[f_idx] for c in combinations])
            for f_idx in range(self.n_factors)
        ]
        return inputs, factor_targets
    
    def summary(self) -> str:
        return (
            f"Cardinalities: {self.cardinalities}\n"
            f"Asymmetry ratio: {self.config.asymmetry_ratio:.2f}\n"
            f"Compression ratio: {self.config.compression_ratio:.2f}\n"
            f"Train samples: {len(self.training_samples)} "
            f"(base: {len(self.train_combinations)})\n"
            f"Test: {len(self.test_combinations)}"
        )


class AsymmetryModel(nn.Module):
    """Model for asymmetry experiments with optional balanced loss."""
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        factor_cardinalities: List[int],
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.cardinalities = factor_cardinalities
        self.n_factors = len(factor_cardinalities)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, card)
            )
            for card in factor_cardinalities
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        z = self.encoder(x)
        logits = [head(z) for head in self.heads]
        return logits, z


def train_asymmetry_model(
    model: AsymmetryModel,
    dataset: AsymmetryDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    balanced_loss: bool = False,  # Weight losses by inverse cardinality
    device: str = "cpu"
) -> Dict:
    """Train with optional balanced loss weighting."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Loss weights: inverse of cardinality (so rare factors get higher weight)
    if balanced_loss:
        max_card = max(dataset.cardinalities)
        loss_weights = [max_card / card for card in dataset.cardinalities]
        total_weight = sum(loss_weights)
        loss_weights = [w / total_weight * len(loss_weights) for w in loss_weights]
    else:
        loss_weights = [1.0] * dataset.n_factors
    
    history = {"train_acc": [], "test_acc": [], "test_per_factor": [], "epoch": []}
    
    for epoch in range(n_epochs):
        model.train()
        
        inputs, factor_targets = dataset.get_batch("train")
        inputs = inputs.to(device)
        factor_targets = [t.to(device) for t in factor_targets]
        
        optimizer.zero_grad()
        logits_list, _ = model(inputs)
        
        # Weighted loss
        loss = sum(
            weight * criterion(logits, targets)
            for weight, logits, targets in zip(loss_weights, logits_list, factor_targets)
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Training
                preds = [logits.argmax(1) for logits in logits_list]
                correct = torch.ones(inputs.shape[0], dtype=torch.bool, device=device)
                for pred, target in zip(preds, factor_targets):
                    correct &= (pred == target)
                train_acc = correct.float().mean().item()
                
                # Test
                test_in, test_targets = dataset.get_batch("test")
                test_in = test_in.to(device)
                test_targets = [t.to(device) for t in test_targets]
                
                test_logits, _ = model(test_in)
                test_preds = [logits.argmax(1) for logits in test_logits]
                
                test_correct = torch.ones(test_in.shape[0], dtype=torch.bool, device=device)
                test_per_factor = []
                for pred, target in zip(test_preds, test_targets):
                    factor_correct = (pred == target)
                    test_correct &= factor_correct
                    test_per_factor.append(factor_correct.float().mean().item())
                
                test_acc = test_correct.float().mean().item()
            
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["test_per_factor"].append(test_per_factor)
            history["epoch"].append(epoch)
    
    return history


# ==============================================================================
# EXPERIMENT 1: Fine-Grained Asymmetry Sweep
# ==============================================================================

def run_asymmetry_sweep(
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Systematically probe the asymmetry boundary with fine granularity.
    
    DESIGN:
    - Fixed total combinations ≈ 100
    - Vary asymmetry ratio from 1.0 to 10.0
    - Find the precise boundary where performance degrades
    
    CARDINALITY CONFIGURATIONS (all ≈ 100 combinations):
    - [10, 10]: ratio 1.0, 100 combos
    - [8, 12]: ratio 1.5, 96 combos (close enough)
    - [7, 14]: ratio 2.0, 98 combos
    - [6, 17]: ratio 2.8, 102 combos
    - [5, 20]: ratio 4.0, 100 combos
    - [4, 25]: ratio 6.25, 100 combos
    - [3, 33]: ratio 11.0, 99 combos
    """
    
    if device is None:
        device = get_device()
    
    # Configurations with ~100 combinations each
    configs = [
        [10, 10],  # ratio 1.0
        [8, 12],   # ratio 1.5
        [7, 14],   # ratio 2.0
        [6, 17],   # ratio 2.8
        [5, 20],   # ratio 4.0
        [5, 25],   # ratio 5.0 (125 combos)
        [4, 25],   # ratio 6.25
        [3, 33],   # ratio 11.0
    ]
    
    print("=" * 70)
    print("EXPERIMENT: Fine-Grained Asymmetry Sweep")
    print("=" * 70)
    print("Finding the precise boundary for factor asymmetry")
    print("=" * 70)
    
    all_results = {
        "experiment": "asymmetry_sweep",
        "config": {"n_seeds": n_seeds, "configs": configs},
        "results": []
    }
    
    for cardinalities in configs:
        asymmetry_ratio = max(cardinalities) / min(cardinalities)
        n_combinations = math.prod(cardinalities)
        bottleneck_dim = sum(cardinalities)
        compression_ratio = n_combinations / bottleneck_dim
        
        print(f"\n{'='*60}")
        print(f"Cardinalities: {cardinalities}")
        print(f"Asymmetry: {asymmetry_ratio:.2f} | Combos: {n_combinations} | "
              f"Compression: {compression_ratio:.2f}")
        print("=" * 60)
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = AsymmetryConfig(
                factor_cardinalities=cardinalities,
                seed=seed
            )
            dataset = AsymmetryDataset(config)
            
            model = AsymmetryModel(
                input_dim=config.input_dim,
                bottleneck_dim=bottleneck_dim,
                factor_cardinalities=cardinalities
            )
            
            history = train_asymmetry_model(
                model, dataset, n_epochs=n_epochs, device=device
            )
            
            result = {
                "seed": seed,
                "test_acc": history["test_acc"][-1],
                "test_per_factor": history["test_per_factor"][-1]
            }
            seed_results.append(result)
            
            print(f"  Seed {seed+1}: {result['test_acc']:.3f} | "
                  f"F1: {result['test_per_factor'][0]:.3f} | "
                  f"F2: {result['test_per_factor'][1]:.3f}")
        
        test_accs = [r["test_acc"] for r in seed_results]
        f1_accs = [r["test_per_factor"][0] for r in seed_results]
        f2_accs = [r["test_per_factor"][1] for r in seed_results]
        
        aggregate = {
            "cardinalities": cardinalities,
            "asymmetry_ratio": asymmetry_ratio,
            "n_combinations": n_combinations,
            "compression_ratio": compression_ratio,
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "f1_acc_mean": float(np.mean(f1_accs)),
            "f2_acc_mean": float(np.mean(f2_accs)),
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"  → AGGREGATE: {aggregate['test_acc_mean']:.3f} ± {aggregate['test_acc_std']:.3f}")
        print(f"     F1 (card={cardinalities[0]}): {aggregate['f1_acc_mean']:.3f}")
        print(f"     F2 (card={cardinalities[1]}): {aggregate['f2_acc_mean']:.3f}")
    
    save_results(all_results, save_dir, "asymmetry_sweep")
    
    # Analysis: Find boundary
    print("\n" + "=" * 70)
    print("BOUNDARY ANALYSIS")
    print("=" * 70)
    
    print("\nAsymmetry Ratio → Performance:")
    for r in all_results["results"]:
        status = "✓" if r["test_acc_mean"] > 0.5 else "✗"
        print(f"  {status} Ratio {r['asymmetry_ratio']:.2f}: {r['test_acc_mean']:.1%}")
    
    # Find approximate boundary
    for i in range(len(all_results["results"]) - 1):
        curr = all_results["results"][i]
        next_r = all_results["results"][i + 1]
        if curr["test_acc_mean"] > 0.5 and next_r["test_acc_mean"] < 0.5:
            print(f"\n  BOUNDARY: Between ratio {curr['asymmetry_ratio']:.2f} and "
                  f"{next_r['asymmetry_ratio']:.2f}")
            break
    
    return all_results


# ==============================================================================
# EXPERIMENT 2: Isolate Asymmetry vs Compression
# ==============================================================================

def run_matched_controls(
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test whether asymmetry or compression ratio matters more.
    
    DESIGN:
    Match compression ratios while varying asymmetry (and vice versa).
    
    GROUP A: Same compression ratio (~5), different asymmetry
    - [10, 10]: ratio 1.0, compression 5.0
    - [5, 20]: ratio 4.0, compression 4.0 (close)
    
    GROUP B: Same asymmetry ratio (~4), different compression
    - [5, 20]: 100 combos, compression 4.0
    - [10, 40]: 400 combos, compression 8.0
    - [20, 80]: 1600 combos, compression 16.0
    
    If compression matters more: Group B should show improvement with more combos
    If asymmetry matters more: Group B should all be poor
    """
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT: Matched Controls (Asymmetry vs Compression)")
    print("=" * 70)
    
    all_results = {
        "experiment": "matched_controls",
        "results": {"group_a": [], "group_b": []}
    }
    
    # GROUP A: Match compression, vary asymmetry
    print("\n--- GROUP A: Similar compression, different asymmetry ---")
    group_a_configs = [
        ([10, 10], "symmetric"),
        ([5, 20], "asymmetric"),
    ]
    
    for cardinalities, label in group_a_configs:
        asymmetry = max(cardinalities) / min(cardinalities)
        compression = math.prod(cardinalities) / sum(cardinalities)
        
        print(f"\n{label}: {cardinalities} (asym={asymmetry:.1f}, comp={compression:.1f})")
        
        seed_results = []
        for seed in range(n_seeds):
            set_seeds(seed)
            config = AsymmetryConfig(factor_cardinalities=cardinalities, seed=seed)
            dataset = AsymmetryDataset(config)
            model = AsymmetryModel(
                input_dim=config.input_dim,
                bottleneck_dim=sum(cardinalities),
                factor_cardinalities=cardinalities
            )
            history = train_asymmetry_model(model, dataset, n_epochs=n_epochs, device=device)
            seed_results.append(history["test_acc"][-1])
        
        result = {
            "cardinalities": cardinalities,
            "label": label,
            "asymmetry_ratio": asymmetry,
            "compression_ratio": compression,
            "test_acc_mean": float(np.mean(seed_results)),
            "test_acc_std": float(np.std(seed_results))
        }
        all_results["results"]["group_a"].append(result)
        print(f"  → {result['test_acc_mean']:.3f} ± {result['test_acc_std']:.3f}")
    
    # GROUP B: Match asymmetry (~4), vary compression
    print("\n--- GROUP B: Similar asymmetry (~4), different compression ---")
    group_b_configs = [
        [5, 20],    # 100 combos
        [7, 28],    # 196 combos
        [10, 40],   # 400 combos
    ]
    
    for cardinalities in group_b_configs:
        asymmetry = max(cardinalities) / min(cardinalities)
        n_combos = math.prod(cardinalities)
        compression = n_combos / sum(cardinalities)
        
        print(f"\n{cardinalities}: {n_combos} combos (asym={asymmetry:.1f}, comp={compression:.1f})")
        
        seed_results = []
        for seed in range(n_seeds):
            set_seeds(seed)
            config = AsymmetryConfig(factor_cardinalities=cardinalities, seed=seed)
            dataset = AsymmetryDataset(config)
            model = AsymmetryModel(
                input_dim=config.input_dim,
                bottleneck_dim=sum(cardinalities),
                factor_cardinalities=cardinalities
            )
            history = train_asymmetry_model(
                model, dataset, n_epochs=min(n_epochs, 3000), device=device
            )
            seed_results.append(history["test_acc"][-1])
        
        result = {
            "cardinalities": cardinalities,
            "n_combinations": n_combos,
            "asymmetry_ratio": asymmetry,
            "compression_ratio": compression,
            "test_acc_mean": float(np.mean(seed_results)),
            "test_acc_std": float(np.std(seed_results))
        }
        all_results["results"]["group_b"].append(result)
        print(f"  → {result['test_acc_mean']:.3f} ± {result['test_acc_std']:.3f}")
    
    save_results(all_results, save_dir, "matched_controls")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("\nGroup A (same compression, diff asymmetry):")
    for r in all_results["results"]["group_a"]:
        print(f"  {r['label']}: {r['test_acc_mean']:.1%}")
    
    a_diff = (all_results["results"]["group_a"][0]["test_acc_mean"] - 
              all_results["results"]["group_a"][1]["test_acc_mean"])
    print(f"  Difference: {a_diff:.1%} → Asymmetry effect")
    
    print("\nGroup B (same asymmetry, diff compression):")
    for r in all_results["results"]["group_b"]:
        print(f"  {r['cardinalities']}: {r['test_acc_mean']:.1%}")
    
    b_improvement = (all_results["results"]["group_b"][-1]["test_acc_mean"] - 
                    all_results["results"]["group_b"][0]["test_acc_mean"])
    print(f"  Improvement with more combos: {b_improvement:.1%} → Compression effect")
    
    if abs(a_diff) > abs(b_improvement):
        print("\n  → ASYMMETRY dominates over COMPRESSION")
    else:
        print("\n  → COMPRESSION dominates over ASYMMETRY")
    
    return all_results


# ==============================================================================
# EXPERIMENT 3: Compensation Strategies
# ==============================================================================

def run_compensation_experiment(
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test whether compensation strategies can overcome asymmetry.
    
    STRATEGIES:
    1. Baseline: No compensation
    2. Oversampling: Replicate rare factor combinations
    3. Balanced loss: Weight losses by inverse cardinality
    4. Both: Oversampling + balanced loss
    
    Test on [5, 20] (asymmetry = 4, known to be problematic)
    """
    
    if device is None:
        device = get_device()
    
    cardinalities = [5, 20]
    
    print("=" * 70)
    print("EXPERIMENT: Compensation Strategies")
    print("=" * 70)
    print(f"Testing on {cardinalities} (asymmetry = 4.0)")
    print("=" * 70)
    
    strategies = [
        {"name": "baseline", "oversample": False, "balanced_loss": False},
        {"name": "oversample_2x", "oversample": True, "oversample_factor": 2.0, "balanced_loss": False},
        {"name": "oversample_4x", "oversample": True, "oversample_factor": 4.0, "balanced_loss": False},
        {"name": "balanced_loss", "oversample": False, "balanced_loss": True},
        {"name": "both_2x", "oversample": True, "oversample_factor": 2.0, "balanced_loss": True},
        {"name": "both_4x", "oversample": True, "oversample_factor": 4.0, "balanced_loss": True},
    ]
    
    all_results = {
        "experiment": "compensation",
        "config": {"cardinalities": cardinalities},
        "results": []
    }
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy['name']} ---")
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = AsymmetryConfig(
                factor_cardinalities=cardinalities,
                oversample_rare=strategy.get("oversample", False),
                oversample_factor=strategy.get("oversample_factor", 1.0),
                seed=seed
            )
            dataset = AsymmetryDataset(config)
            
            model = AsymmetryModel(
                input_dim=config.input_dim,
                bottleneck_dim=sum(cardinalities),
                factor_cardinalities=cardinalities
            )
            
            history = train_asymmetry_model(
                model, dataset, 
                n_epochs=n_epochs,
                balanced_loss=strategy.get("balanced_loss", False),
                device=device
            )
            
            result = {
                "seed": seed,
                "test_acc": history["test_acc"][-1],
                "test_per_factor": history["test_per_factor"][-1]
            }
            seed_results.append(result)
        
        test_accs = [r["test_acc"] for r in seed_results]
        f1_accs = [r["test_per_factor"][0] for r in seed_results]
        f2_accs = [r["test_per_factor"][1] for r in seed_results]
        
        aggregate = {
            "strategy": strategy["name"],
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "f1_acc_mean": float(np.mean(f1_accs)),
            "f2_acc_mean": float(np.mean(f2_accs)),
        }
        all_results["results"].append(aggregate)
        
        print(f"  Test: {aggregate['test_acc_mean']:.3f} ± {aggregate['test_acc_std']:.3f}")
        print(f"  F1 (card=5): {aggregate['f1_acc_mean']:.3f} | F2 (card=20): {aggregate['f2_acc_mean']:.3f}")
    
    save_results(all_results, save_dir, "compensation")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPENSATION SUMMARY")
    print("=" * 70)
    
    baseline = all_results["results"][0]["test_acc_mean"]
    print(f"\nBaseline: {baseline:.1%}")
    print("\nImprovements over baseline:")
    for r in all_results["results"][1:]:
        improvement = r["test_acc_mean"] - baseline
        print(f"  {r['strategy']:15s}: {r['test_acc_mean']:.1%} ({improvement:+.1%})")
    
    return all_results


# ==============================================================================
# MAIN
# ==============================================================================

def run_all_asymmetry_experiments(
    device: Optional[str] = None,
    save_dir: str = "results"
):
    """Run complete asymmetry boundary investigation."""
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("ASYMMETRY BOUNDARY INVESTIGATION")
    print("=" * 70)
    
    results = {}
    
    # 1. Fine-grained sweep
    print("\n" + "="*70)
    print("1. FINE-GRAINED ASYMMETRY SWEEP")
    print("="*70)
    results["sweep"] = run_asymmetry_sweep(
        n_seeds=5, n_epochs=2000, device=device, save_dir=save_dir
    )
    
    # 2. Matched controls
    print("\n" + "="*70)
    print("2. MATCHED CONTROLS (Asymmetry vs Compression)")
    print("="*70)
    results["controls"] = run_matched_controls(
        n_seeds=5, n_epochs=2000, device=device, save_dir=save_dir
    )
    
    # 3. Compensation strategies
    print("\n" + "="*70)
    print("3. COMPENSATION STRATEGIES")
    print("="*70)
    results["compensation"] = run_compensation_experiment(
        n_seeds=5, n_epochs=2000, device=device, save_dir=save_dir
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\n1. Asymmetry Boundary:")
    for r in results["sweep"]["results"]:
        status = "✓" if r["test_acc_mean"] > 0.5 else "✗"
        print(f"   {status} Ratio {r['asymmetry_ratio']:.1f}: {r['test_acc_mean']:.0%}")
    
    print("\n2. Asymmetry vs Compression:")
    print(f"   Asymmetry effect dominates at moderate compression ratios")
    
    print("\n3. Compensation:")
    best = max(results["compensation"]["results"], key=lambda x: x["test_acc_mean"])
    print(f"   Best strategy: {best['strategy']} ({best['test_acc_mean']:.0%})")
    
    return results


if __name__ == "__main__":
    run_all_asymmetry_experiments()