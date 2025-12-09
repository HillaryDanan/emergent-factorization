"""
EXTENDED EXPERIMENTS: Robustness and Generalization Tests

Building on the core finding (End-to-End Compositionality Principle),
these experiments probe its boundaries and robustness:

1. SCALING: 4, 5, 6+ factors - does perfect generalization continue?
2. ASYMMETRY: Varying factor cardinalities - does symmetry matter?
3. NOISE: Label noise, input noise - how robust is the finding?
4. ARCHITECTURE: Transformers, CNNs, RNNs - does this generalize beyond MLPs?

Design principles:
- Systematic parameter sweeps
- Matched controls
- Statistical rigor (multiple seeds, effect sizes)
- Clear falsification criteria

References:
- Keysers et al. (2020), ICLR - comprehensive compositional generalization
- Bahdanau et al. (2019), ICLR - systematic generalization
- Lake & Baroni (2018), ICML - SCAN benchmark
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import product
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seeds, get_device, save_results


# ==============================================================================
# FLEXIBLE N-FACTOR DATASET
# ==============================================================================

@dataclass
class NFactorConfig:
    """Configuration for N-factor factorial dataset."""
    factor_cardinalities: List[int] = field(default_factory=lambda: [10, 10])
    holdout_fraction: float = 0.25
    input_noise_std: float = 0.0  # Gaussian noise on inputs
    label_noise_prob: float = 0.0  # Probability of random label
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


class NFactorDataset:
    """
    Flexible N-factor factorial dataset.
    
    Supports:
    - Arbitrary number of factors
    - Arbitrary cardinalities per factor (asymmetric OK)
    - Input noise (Gaussian)
    - Label noise (random corruption)
    
    Input representation: Concatenated one-hots (factorized)
    """
    
    def __init__(self, config: NFactorConfig):
        self.config = config
        self.cardinalities = config.factor_cardinalities
        self.n_factors = config.n_factors
        self.n_combinations = config.n_combinations
        self.input_dim = config.input_dim
        
        np.random.seed(config.seed)
        
        # Generate all combinations as tuples
        self.all_combinations = list(product(*[range(c) for c in self.cardinalities]))
        
        self._create_split()
    
    def _create_split(self):
        """Compositional holdout ensuring all factor values appear in training."""
        n_holdout = int(self.n_combinations * self.config.holdout_fraction)
        
        # Track factor value occurrences
        factor_counts = [defaultdict(int) for _ in range(self.n_factors)]
        for combo in self.all_combinations:
            for f_idx, f_val in enumerate(combo):
                factor_counts[f_idx][f_val] += 1
        
        # Find candidates where all factor values appear elsewhere
        candidates = []
        for combo in self.all_combinations:
            can_holdout = all(
                factor_counts[f_idx][f_val] > 1
                for f_idx, f_val in enumerate(combo)
            )
            if can_holdout:
                candidates.append(combo)
        
        np.random.shuffle(candidates)
        
        # Select holdout
        self.test_combinations = []
        temp_counts = [fc.copy() for fc in factor_counts]
        
        for combo in candidates:
            if len(self.test_combinations) >= n_holdout:
                break
            can_take = all(
                temp_counts[f_idx][f_val] > 1
                for f_idx, f_val in enumerate(combo)
            )
            if can_take:
                self.test_combinations.append(combo)
                for f_idx, f_val in enumerate(combo):
                    temp_counts[f_idx][f_val] -= 1
        
        self.train_combinations = [
            c for c in self.all_combinations
            if c not in self.test_combinations
        ]
    
    def get_input(self, combination: Tuple) -> torch.Tensor:
        """Factorized input: concatenated one-hots."""
        parts = []
        for f_idx, f_val in enumerate(combination):
            one_hot = torch.zeros(self.cardinalities[f_idx])
            one_hot[f_val] = 1.0
            parts.append(one_hot)
        
        x = torch.cat(parts)
        
        # Add input noise if configured
        if self.config.input_noise_std > 0:
            x = x + torch.randn_like(x) * self.config.input_noise_std
        
        return x
    
    def get_factor_targets(self, combination: Tuple) -> List[torch.Tensor]:
        """Get target for each factor."""
        targets = [torch.tensor(f_val) for f_val in combination]
        
        # Add label noise if configured
        if self.config.label_noise_prob > 0:
            for f_idx in range(len(targets)):
                if np.random.random() < self.config.label_noise_prob:
                    # Random label
                    targets[f_idx] = torch.tensor(
                        np.random.randint(0, self.cardinalities[f_idx])
                    )
        
        return targets
    
    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get batch: (inputs, [factor_targets])."""
        combinations = self.train_combinations if split == "train" else self.test_combinations
        
        inputs = torch.stack([self.get_input(c) for c in combinations])
        
        # List of tensors, one per factor
        factor_targets = [
            torch.stack([self.get_factor_targets(c)[f_idx] for c in combinations])
            for f_idx in range(self.n_factors)
        ]
        
        return inputs, factor_targets
    
    def summary(self) -> str:
        return (
            f"{self.n_factors}-Factor Dataset: "
            f"{' × '.join(map(str, self.cardinalities))} = {self.n_combinations}\n"
            f"Input dim: {self.input_dim}\n"
            f"Train: {len(self.train_combinations)} | Test: {len(self.test_combinations)}\n"
            f"Noise: input_std={self.config.input_noise_std}, "
            f"label_prob={self.config.label_noise_prob}"
        )


# ==============================================================================
# FLEXIBLE N-FACTOR MODEL
# ==============================================================================

class NFactorCompositionalModel(nn.Module):
    """
    Compositional model for N factors.
    
    Architecture:
        Input → Encoder → Bottleneck → [Head_1, Head_2, ..., Head_N]
    
    Each head predicts one factor independently.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        factor_cardinalities: List[int],
        hidden_dim: int = 128,
        encoder_type: str = "mlp"  # "mlp", "transformer", "rnn"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.cardinalities = factor_cardinalities
        self.n_factors = len(factor_cardinalities)
        self.encoder_type = encoder_type
        
        # Build encoder based on type
        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, bottleneck_dim)
            )
        elif encoder_type == "transformer":
            # Simple transformer-style encoder
            # Treat input as sequence of factor encodings
            self.factor_embedding = nn.Linear(max(factor_cardinalities), hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.bottleneck_proj = nn.Linear(hidden_dim * len(factor_cardinalities), bottleneck_dim)
        elif encoder_type == "rnn":
            self.rnn = nn.LSTM(
                input_size=max(factor_cardinalities),
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True
            )
            self.bottleneck_proj = nn.Linear(hidden_dim, bottleneck_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Build heads (one per factor)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, cardinality)
            )
            for cardinality in factor_cardinalities
        ])
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to bottleneck."""
        if self.encoder_type == "mlp":
            return self.encoder(x)
        
        elif self.encoder_type == "transformer":
            # Reshape to sequence of factor encodings
            batch_size = x.shape[0]
            # Split input by factors and pad to max cardinality
            max_card = max(self.cardinalities)
            factor_inputs = []
            offset = 0
            for card in self.cardinalities:
                factor_onehot = x[:, offset:offset+card]
                # Pad to max cardinality
                if card < max_card:
                    padding = torch.zeros(batch_size, max_card - card, device=x.device)
                    factor_onehot = torch.cat([factor_onehot, padding], dim=1)
                factor_inputs.append(factor_onehot)
                offset += card
            
            # Stack as sequence: [batch, n_factors, max_card]
            seq = torch.stack(factor_inputs, dim=1)
            # Embed
            seq = self.factor_embedding(seq)  # [batch, n_factors, hidden]
            # Transform
            seq = self.transformer(seq)  # [batch, n_factors, hidden]
            # Flatten and project
            flat = seq.reshape(batch_size, -1)
            return self.bottleneck_proj(flat)
        
        elif self.encoder_type == "rnn":
            # Similar reshaping
            batch_size = x.shape[0]
            max_card = max(self.cardinalities)
            factor_inputs = []
            offset = 0
            for card in self.cardinalities:
                factor_onehot = x[:, offset:offset+card]
                if card < max_card:
                    padding = torch.zeros(batch_size, max_card - card, device=x.device)
                    factor_onehot = torch.cat([factor_onehot, padding], dim=1)
                factor_inputs.append(factor_onehot)
                offset += card
            
            seq = torch.stack(factor_inputs, dim=1)  # [batch, n_factors, max_card]
            _, (h_n, _) = self.rnn(seq)
            return self.bottleneck_proj(h_n[-1])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            logits: List of logits per factor
            bottleneck: Bottleneck representation
        """
        z = self.encode(x)
        logits = [head(z) for head in self.heads]
        return logits, z


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_nfactor_model(
    model: NFactorCompositionalModel,
    dataset: NFactorDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> Dict:
    """Train N-factor model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_acc_all": [], "test_acc_all": [],
        "test_acc_per_factor": [[] for _ in range(dataset.n_factors)],
        "epoch": []
    }
    
    for epoch in range(n_epochs):
        model.train()
        
        inputs, factor_targets = dataset.get_batch("train")
        inputs = inputs.to(device)
        factor_targets = [t.to(device) for t in factor_targets]
        
        optimizer.zero_grad()
        logits_list, _ = model(inputs)
        
        # Sum losses across factors
        loss = sum(
            criterion(logits, targets)
            for logits, targets in zip(logits_list, factor_targets)
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Training accuracy (all factors correct)
                preds = [logits.argmax(1) for logits in logits_list]
                correct_all = torch.ones(inputs.shape[0], dtype=torch.bool, device=device)
                for pred, target in zip(preds, factor_targets):
                    correct_all &= (pred == target)
                train_acc_all = correct_all.float().mean().item()
                
                # Test accuracy
                test_inputs, test_targets = dataset.get_batch("test")
                test_inputs = test_inputs.to(device)
                test_targets = [t.to(device) for t in test_targets]
                
                test_logits, _ = model(test_inputs)
                test_preds = [logits.argmax(1) for logits in test_logits]
                
                correct_all_test = torch.ones(test_inputs.shape[0], dtype=torch.bool, device=device)
                test_acc_per = []
                for f_idx, (pred, target) in enumerate(zip(test_preds, test_targets)):
                    factor_correct = (pred == target)
                    correct_all_test &= factor_correct
                    test_acc_per.append(factor_correct.float().mean().item())
                
                test_acc_all = correct_all_test.float().mean().item()
            
            history["train_acc_all"].append(train_acc_all)
            history["test_acc_all"].append(test_acc_all)
            for f_idx, acc in enumerate(test_acc_per):
                history["test_acc_per_factor"][f_idx].append(acc)
            history["epoch"].append(epoch)
    
    return history


# ==============================================================================
# EXPERIMENT 1: SCALING (4, 5, 6+ FACTORS)
# ==============================================================================

def run_scaling_experiment(
    n_factors_list: List[int] = [2, 3, 4, 5, 6],
    factor_cardinality: int = 4,  # Keep cardinality small for tractability
    n_seeds: int = 5,
    n_epochs: int = 3000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test how compositional generalization scales with number of factors.
    
    PREDICTION:
    Based on 2-factor (80%) vs 3-factor (100%) results, we predict:
    - Higher factors should maintain high generalization
    - May see ceiling effect at 100%
    - OR may eventually see degradation due to optimization difficulty
    
    DESIGN:
    - Fixed cardinality per factor (for fair comparison)
    - Bottleneck at theoretical threshold (sum of cardinalities)
    """
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT: SCALING TO N FACTORS")
    print("=" * 70)
    print(f"Testing n_factors: {n_factors_list}")
    print(f"Factor cardinality: {factor_cardinality} each")
    print("=" * 70)
    
    all_results = {
        "experiment": "scaling_n_factors",
        "config": {
            "n_factors_list": n_factors_list,
            "factor_cardinality": factor_cardinality,
            "n_seeds": n_seeds
        },
        "results": []
    }
    
    for n_factors in n_factors_list:
        cardinalities = [factor_cardinality] * n_factors
        n_combinations = factor_cardinality ** n_factors
        bottleneck_dim = sum(cardinalities)  # Theoretical threshold
        
        print(f"\n{'='*60}")
        print(f"N_FACTORS: {n_factors}")
        print(f"Combinations: {n_combinations}")
        print(f"Bottleneck: {bottleneck_dim}")
        print("=" * 60)
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = NFactorConfig(
                factor_cardinalities=cardinalities,
                holdout_fraction=0.25,
                seed=seed
            )
            dataset = NFactorDataset(config)
            
            model = NFactorCompositionalModel(
                input_dim=config.input_dim,
                bottleneck_dim=bottleneck_dim,
                factor_cardinalities=cardinalities
            )
            
            print(f"  Seed {seed+1}/{n_seeds}...", end=" ")
            history = train_nfactor_model(
                model, dataset, n_epochs=n_epochs, device=device
            )
            
            result = {
                "seed": seed,
                "train_acc": history["train_acc_all"][-1],
                "test_acc": history["test_acc_all"][-1],
                "test_per_factor": [h[-1] for h in history["test_acc_per_factor"]]
            }
            seed_results.append(result)
            
            print(f"Train: {result['train_acc']:.3f} | Test: {result['test_acc']:.3f}")
        
        test_accs = [r["test_acc"] for r in seed_results]
        aggregate = {
            "n_factors": n_factors,
            "n_combinations": n_combinations,
            "bottleneck_dim": bottleneck_dim,
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"  → AGGREGATE: {aggregate['test_acc_mean']:.3f} ± {aggregate['test_acc_std']:.3f}")
    
    save_results(all_results, save_dir, "scaling_n_factors")
    return all_results


# ==============================================================================
# EXPERIMENT 2: ASYMMETRIC CARDINALITIES
# ==============================================================================

def run_asymmetry_experiment(
    cardinality_configs: List[List[int]] = None,
    n_seeds: int = 5,
    n_epochs: int = 3000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test whether factor asymmetry affects compositional generalization.
    
    QUESTION:
    Does it matter if factors have different cardinalities?
    e.g., [10, 10] vs [5, 20] vs [2, 50] (all = 100 combos)
    
    PREDICTION:
    Symmetric may be easier (balanced representation) OR
    Asymmetric may be fine (compositional structure is key)
    """
    
    if cardinality_configs is None:
        # Same total combinations, different distributions
        cardinality_configs = [
            [10, 10],       # Symmetric: 100 combos
            [5, 20],        # Mild asymmetry: 100 combos
            [4, 25],        # Moderate asymmetry: 100 combos
            [2, 50],        # Extreme asymmetry: 100 combos
            [10, 10, 1],    # 3 factors with degenerate: 100 combos
            [5, 5, 4],      # 3 balanced factors: 100 combos
        ]
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT: ASYMMETRIC CARDINALITIES")
    print("=" * 70)
    
    all_results = {
        "experiment": "asymmetric_cardinalities",
        "config": {"cardinality_configs": cardinality_configs, "n_seeds": n_seeds},
        "results": []
    }
    
    for cardinalities in cardinality_configs:
        n_combinations = math.prod(cardinalities)
        bottleneck_dim = sum(cardinalities)
        
        print(f"\n{'='*60}")
        print(f"Cardinalities: {cardinalities} = {n_combinations} combinations")
        print(f"Bottleneck: {bottleneck_dim}")
        print("=" * 60)
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = NFactorConfig(
                factor_cardinalities=cardinalities,
                holdout_fraction=0.25,
                seed=seed
            )
            dataset = NFactorDataset(config)
            
            model = NFactorCompositionalModel(
                input_dim=config.input_dim,
                bottleneck_dim=bottleneck_dim,
                factor_cardinalities=cardinalities
            )
            
            print(f"  Seed {seed+1}/{n_seeds}...", end=" ")
            history = train_nfactor_model(
                model, dataset, n_epochs=n_epochs, device=device
            )
            
            result = {
                "seed": seed,
                "test_acc": history["test_acc_all"][-1]
            }
            seed_results.append(result)
            print(f"Test: {result['test_acc']:.3f}")
        
        test_accs = [r["test_acc"] for r in seed_results]
        aggregate = {
            "cardinalities": cardinalities,
            "n_combinations": n_combinations,
            "bottleneck_dim": bottleneck_dim,
            "asymmetry_ratio": max(cardinalities) / min(cardinalities),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"  → AGGREGATE: {aggregate['test_acc_mean']:.3f} ± {aggregate['test_acc_std']:.3f}")
    
    save_results(all_results, save_dir, "asymmetric_cardinalities")
    return all_results


# ==============================================================================
# EXPERIMENT 3: NOISE ROBUSTNESS
# ==============================================================================

def run_noise_experiment(
    input_noise_levels: List[float] = [0.0, 0.1, 0.2, 0.5, 1.0],
    label_noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3],
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test robustness of compositional generalization to noise.
    
    TYPES OF NOISE:
    1. Input noise: Gaussian noise added to one-hot inputs
    2. Label noise: Random label corruption during training
    
    PREDICTION:
    - Moderate noise should be tolerable
    - High noise should degrade gracefully
    - Label noise may be more harmful than input noise
    """
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT: NOISE ROBUSTNESS")
    print("=" * 70)
    
    all_results = {
        "experiment": "noise_robustness",
        "config": {
            "input_noise_levels": input_noise_levels,
            "label_noise_levels": label_noise_levels,
            "n_seeds": n_seeds
        },
        "results": {"input_noise": [], "label_noise": []}
    }
    
    base_cardinalities = [10, 10]
    bottleneck_dim = sum(base_cardinalities)
    
    # Test input noise
    print("\n--- INPUT NOISE ---")
    for noise_std in input_noise_levels:
        print(f"\nInput noise std: {noise_std}")
        
        seed_results = []
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = NFactorConfig(
                factor_cardinalities=base_cardinalities,
                input_noise_std=noise_std,
                seed=seed
            )
            dataset = NFactorDataset(config)
            
            model = NFactorCompositionalModel(
                input_dim=config.input_dim,
                bottleneck_dim=bottleneck_dim,
                factor_cardinalities=base_cardinalities
            )
            
            history = train_nfactor_model(model, dataset, n_epochs=n_epochs, device=device)
            seed_results.append({"test_acc": history["test_acc_all"][-1]})
        
        test_accs = [r["test_acc"] for r in seed_results]
        all_results["results"]["input_noise"].append({
            "noise_std": noise_std,
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs))
        })
        print(f"  Test: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    
    # Test label noise
    print("\n--- LABEL NOISE ---")
    for noise_prob in label_noise_levels:
        print(f"\nLabel noise prob: {noise_prob}")
        
        seed_results = []
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = NFactorConfig(
                factor_cardinalities=base_cardinalities,
                label_noise_prob=noise_prob,
                seed=seed
            )
            dataset = NFactorDataset(config)
            
            model = NFactorCompositionalModel(
                input_dim=config.input_dim,
                bottleneck_dim=bottleneck_dim,
                factor_cardinalities=base_cardinalities
            )
            
            history = train_nfactor_model(model, dataset, n_epochs=n_epochs, device=device)
            seed_results.append({"test_acc": history["test_acc_all"][-1]})
        
        test_accs = [r["test_acc"] for r in seed_results]
        all_results["results"]["label_noise"].append({
            "noise_prob": noise_prob,
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs))
        })
        print(f"  Test: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    
    save_results(all_results, save_dir, "noise_robustness")
    return all_results


# ==============================================================================
# EXPERIMENT 4: ARCHITECTURE COMPARISON
# ==============================================================================

def run_architecture_experiment(
    architectures: List[str] = ["mlp", "transformer", "rnn"],
    bottleneck_sizes: List[int] = [10, 20, 30],
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test whether the finding generalizes across architectures.
    
    ARCHITECTURES:
    - MLP: Feed-forward (baseline)
    - Transformer: Self-attention over factor encodings
    - RNN: Sequential processing of factor encodings
    
    PREDICTION:
    End-to-End Compositionality Principle should hold across architectures.
    The key is output structure, not encoder architecture.
    """
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT: ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    all_results = {
        "experiment": "architecture_comparison",
        "config": {
            "architectures": architectures,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds
        },
        "results": []
    }
    
    cardinalities = [10, 10]
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"ARCHITECTURE: {arch.upper()}")
        print("=" * 60)
        
        for bottleneck_dim in bottleneck_sizes:
            print(f"\n  Bottleneck: {bottleneck_dim}")
            
            seed_results = []
            for seed in range(n_seeds):
                set_seeds(seed)
                
                config = NFactorConfig(
                    factor_cardinalities=cardinalities,
                    seed=seed
                )
                dataset = NFactorDataset(config)
                
                try:
                    model = NFactorCompositionalModel(
                        input_dim=config.input_dim,
                        bottleneck_dim=bottleneck_dim,
                        factor_cardinalities=cardinalities,
                        encoder_type=arch
                    )
                    
                    history = train_nfactor_model(
                        model, dataset, n_epochs=n_epochs, device=device
                    )
                    test_acc = history["test_acc_all"][-1]
                except Exception as e:
                    print(f"    Error with {arch}: {e}")
                    test_acc = 0.0
                
                seed_results.append({"test_acc": test_acc})
            
            test_accs = [r["test_acc"] for r in seed_results]
            result = {
                "architecture": arch,
                "bottleneck_dim": bottleneck_dim,
                "test_acc_mean": float(np.mean(test_accs)),
                "test_acc_std": float(np.std(test_accs))
            }
            all_results["results"].append(result)
            
            print(f"    Test: {result['test_acc_mean']:.3f} ± {result['test_acc_std']:.3f}")
    
    save_results(all_results, save_dir, "architecture_comparison")
    return all_results


# ==============================================================================
# MAIN: RUN ALL EXTENDED EXPERIMENTS
# ==============================================================================

def run_all_extended_experiments(
    device: Optional[str] = None,
    save_dir: str = "results"
):
    """Run complete extended experimental suite."""
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXTENDED EXPERIMENTS: ROBUSTNESS & GENERALIZATION")
    print("=" * 70)
    
    results = {}
    
    # 1. Scaling
    print("\n" + "="*70)
    print("1. SCALING TO N FACTORS")
    print("="*70)
    results["scaling"] = run_scaling_experiment(
        n_factors_list=[2, 3, 4, 5],
        factor_cardinality=4,
        n_seeds=5,
        n_epochs=3000,
        device=device,
        save_dir=save_dir
    )
    
    # 2. Asymmetry
    print("\n" + "="*70)
    print("2. ASYMMETRIC CARDINALITIES")
    print("="*70)
    results["asymmetry"] = run_asymmetry_experiment(
        n_seeds=5,
        n_epochs=2000,
        device=device,
        save_dir=save_dir
    )
    
    # 3. Noise
    print("\n" + "="*70)
    print("3. NOISE ROBUSTNESS")
    print("="*70)
    results["noise"] = run_noise_experiment(
        n_seeds=5,
        n_epochs=2000,
        device=device,
        save_dir=save_dir
    )
    
    # 4. Architecture
    print("\n" + "="*70)
    print("4. ARCHITECTURE COMPARISON")
    print("="*70)
    results["architecture"] = run_architecture_experiment(
        architectures=["mlp", "transformer", "rnn"],
        bottleneck_sizes=[10, 20, 30],
        n_seeds=5,
        n_epochs=2000,
        device=device,
        save_dir=save_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("EXTENDED EXPERIMENTS SUMMARY")
    print("="*70)
    
    print("\n1. SCALING:")
    for r in results["scaling"]["results"]:
        print(f"   {r['n_factors']} factors: {r['test_acc_mean']:.1%} ± {r['test_acc_std']:.1%}")
    
    print("\n2. ASYMMETRY:")
    for r in results["asymmetry"]["results"]:
        print(f"   {r['cardinalities']}: {r['test_acc_mean']:.1%} (ratio={r['asymmetry_ratio']:.1f})")
    
    print("\n3. NOISE:")
    print("   Input noise:")
    for r in results["noise"]["results"]["input_noise"]:
        print(f"     std={r['noise_std']}: {r['test_acc_mean']:.1%}")
    print("   Label noise:")
    for r in results["noise"]["results"]["label_noise"]:
        print(f"     prob={r['noise_prob']}: {r['test_acc_mean']:.1%}")
    
    print("\n4. ARCHITECTURE (bottleneck=20):")
    for r in results["architecture"]["results"]:
        if r["bottleneck_dim"] == 20:
            print(f"   {r['architecture']:12s}: {r['test_acc_mean']:.1%} ± {r['test_acc_std']:.1%}")
    
    return results


if __name__ == "__main__":
    run_all_extended_experiments()