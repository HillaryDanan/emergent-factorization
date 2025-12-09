"""
EXPERIMENT 6: Three-Factor Extension

MOTIVATION:
Does the compositional output finding scale to more complex factorial structures?

DESIGN:
- 3 factors: Color (8) × Shape (8) × Size (4) = 256 combinations
- Factorized input: 8 + 8 + 4 = 20 dimensions
- Compositional output: 3 separate heads (f1, f2, f3)
- Test: ALL THREE factors correct on novel combinations

PREDICTIONS:
1. If compositional output enables generalization, it should scale
2. Generalization should depend on bottleneck capacity (now need to represent 3 factors)
3. Theoretical threshold: 8 + 8 + 4 = 20 dimensions

COMPLEXITY ANALYSIS:
- 2 factors: Need to generalize across 2D grid
- 3 factors: Need to generalize across 3D cube
- This is a harder test of true compositional generalization

SCIENTIFIC QUESTION:
Does compositional structure scale, or does it only work for simple 2-factor cases?

References:
- Scaling of compositional generalization: Keysers et al. (2020), ICLR
- Systematicity in complex domains: Bahdanau et al. (2019), ICLR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from .utils import set_seeds, get_device, save_results


@dataclass
class ThreeFactorConfig:
    """Configuration for 3-factor dataset."""
    n_factor1: int = 8   # e.g., colors
    n_factor2: int = 8   # e.g., shapes
    n_factor3: int = 4   # e.g., sizes
    holdout_fraction: float = 0.25
    seed: int = 42


class ThreeFactorDataset:
    """
    Dataset with 3 factorial dimensions.
    
    Factorized input: [f1_onehot, f2_onehot, f3_onehot]
    
    Holdout strategy ensures:
    - Each factor value appears in training
    - Held-out combinations are novel 3-way compositions
    """
    
    def __init__(self, config: ThreeFactorConfig):
        self.config = config
        self.n_factor1 = config.n_factor1
        self.n_factor2 = config.n_factor2
        self.n_factor3 = config.n_factor3
        self.n_combinations = config.n_factor1 * config.n_factor2 * config.n_factor3
        self.input_dim = config.n_factor1 + config.n_factor2 + config.n_factor3
        
        np.random.seed(config.seed)
        
        # All combinations as (f1, f2, f3) tuples
        self.all_combinations = [
            (i, j, k)
            for i in range(self.n_factor1)
            for j in range(self.n_factor2)
            for k in range(self.n_factor3)
        ]
        
        self._create_split()
    
    def _create_split(self):
        """Create compositional holdout for 3 factors."""
        n_holdout = int(self.n_combinations * self.config.holdout_fraction)
        
        # Track factor occurrences
        f1_counts = defaultdict(int)
        f2_counts = defaultdict(int)
        f3_counts = defaultdict(int)
        
        for f1, f2, f3 in self.all_combinations:
            f1_counts[f1] += 1
            f2_counts[f2] += 1
            f3_counts[f3] += 1
        
        # Find candidates where all three factors appear elsewhere
        candidates = [
            (f1, f2, f3) for f1, f2, f3 in self.all_combinations
            if f1_counts[f1] > 1 and f2_counts[f2] > 1 and f3_counts[f3] > 1
        ]
        np.random.shuffle(candidates)
        
        self.test_combinations = []
        temp_f1 = f1_counts.copy()
        temp_f2 = f2_counts.copy()
        temp_f3 = f3_counts.copy()
        
        for f1, f2, f3 in candidates:
            if len(self.test_combinations) >= n_holdout:
                break
            if temp_f1[f1] > 1 and temp_f2[f2] > 1 and temp_f3[f3] > 1:
                self.test_combinations.append((f1, f2, f3))
                temp_f1[f1] -= 1
                temp_f2[f2] -= 1
                temp_f3[f3] -= 1
        
        self.train_combinations = [
            c for c in self.all_combinations
            if c not in self.test_combinations
        ]
    
    def get_input(self, combination: Tuple[int, int, int]) -> torch.Tensor:
        """Factorized input: concatenated one-hots for each factor."""
        f1, f2, f3 = combination
        
        f1_onehot = torch.zeros(self.n_factor1)
        f1_onehot[f1] = 1.0
        
        f2_onehot = torch.zeros(self.n_factor2)
        f2_onehot[f2] = 1.0
        
        f3_onehot = torch.zeros(self.n_factor3)
        f3_onehot[f3] = 1.0
        
        return torch.cat([f1_onehot, f2_onehot, f3_onehot])
    
    def get_factor_targets(
        self, combination: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1, f2, f3 = combination
        return torch.tensor(f1), torch.tensor(f2), torch.tensor(f3)
    
    def get_batch(
        self, split: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (inputs, f1_targets, f2_targets, f3_targets)."""
        combinations = self.train_combinations if split == "train" else self.test_combinations
        
        inputs = torch.stack([self.get_input(c) for c in combinations])
        f1 = torch.stack([self.get_factor_targets(c)[0] for c in combinations])
        f2 = torch.stack([self.get_factor_targets(c)[1] for c in combinations])
        f3 = torch.stack([self.get_factor_targets(c)[2] for c in combinations])
        
        return inputs, f1, f2, f3
    
    def summary(self) -> str:
        return (
            f"3-Factor Dataset: {self.n_factor1} × {self.n_factor2} × {self.n_factor3} "
            f"= {self.n_combinations} combinations\n"
            f"Input dim: {self.input_dim}\n"
            f"Train: {len(self.train_combinations)} | Test: {len(self.test_combinations)}"
        )


class ThreeFactorCompositionalModel(nn.Module):
    """
    Model for 3-factor compositional output.
    
    Three separate prediction heads, one for each factor.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        n_factor1: int,
        n_factor2: int,
        n_factor3: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        self.head_f1 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor1)
        )
        
        self.head_f2 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor2)
        )
        
        self.head_f3 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor3)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.head_f1(z), self.head_f2(z), self.head_f3(z), z


def train_three_factor_model(
    model: ThreeFactorCompositionalModel,
    dataset: ThreeFactorDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> Dict:
    """Train 3-factor model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_acc_all": [], "test_acc_all": [],
        "test_acc_f1": [], "test_acc_f2": [], "test_acc_f3": [],
        "epoch": []
    }
    
    for epoch in range(n_epochs):
        model.train()
        
        inputs, f1_targets, f2_targets, f3_targets = dataset.get_batch("train")
        inputs = inputs.to(device)
        f1_targets = f1_targets.to(device)
        f2_targets = f2_targets.to(device)
        f3_targets = f3_targets.to(device)
        
        optimizer.zero_grad()
        logits_f1, logits_f2, logits_f3, _ = model(inputs)
        
        loss = (criterion(logits_f1, f1_targets) + 
                criterion(logits_f2, f2_targets) + 
                criterion(logits_f3, f3_targets))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Train
                pred_f1 = logits_f1.argmax(1)
                pred_f2 = logits_f2.argmax(1)
                pred_f3 = logits_f3.argmax(1)
                
                train_all = (
                    (pred_f1 == f1_targets) & 
                    (pred_f2 == f2_targets) & 
                    (pred_f3 == f3_targets)
                ).float().mean().item()
                
                # Test
                test_in, test_f1, test_f2, test_f3 = dataset.get_batch("test")
                test_in = test_in.to(device)
                test_f1, test_f2, test_f3 = test_f1.to(device), test_f2.to(device), test_f3.to(device)
                
                t_logits_f1, t_logits_f2, t_logits_f3, _ = model(test_in)
                t_pred_f1 = t_logits_f1.argmax(1)
                t_pred_f2 = t_logits_f2.argmax(1)
                t_pred_f3 = t_logits_f3.argmax(1)
                
                test_acc_f1 = (t_pred_f1 == test_f1).float().mean().item()
                test_acc_f2 = (t_pred_f2 == test_f2).float().mean().item()
                test_acc_f3 = (t_pred_f3 == test_f3).float().mean().item()
                test_all = (
                    (t_pred_f1 == test_f1) & 
                    (t_pred_f2 == test_f2) & 
                    (t_pred_f3 == test_f3)
                ).float().mean().item()
            
            history["train_acc_all"].append(train_all)
            history["test_acc_all"].append(test_all)
            history["test_acc_f1"].append(test_acc_f1)
            history["test_acc_f2"].append(test_acc_f2)
            history["test_acc_f3"].append(test_acc_f3)
            history["epoch"].append(epoch)
    
    return history


def run_three_factor_experiment(
    bottleneck_sizes: List[int],
    n_factor1: int = 8,
    n_factor2: int = 8,
    n_factor3: int = 4,
    n_seeds: int = 5,
    n_epochs: int = 3000,  # More epochs for harder task
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Test compositional generalization with 3 factors.
    """
    
    if device is None:
        device = get_device()
    
    n_combinations = n_factor1 * n_factor2 * n_factor3
    input_dim = n_factor1 + n_factor2 + n_factor3
    
    print("=" * 70)
    print("EXPERIMENT 6: THREE-FACTOR EXTENSION")
    print("=" * 70)
    print(f"Factors: {n_factor1} × {n_factor2} × {n_factor3} = {n_combinations} combinations")
    print(f"Input dim: {input_dim} (factorized)")
    print(f"Output: 3 compositional heads")
    print(f"Test: ALL THREE factors correct")
    print("=" * 70)
    
    all_results = {
        "experiment": "three_factors",
        "config": {
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "n_factor3": n_factor3,
            "n_combinations": n_combinations,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds,
            "theoretical_threshold": input_dim
        },
        "results": []
    }
    
    for bottleneck_dim in bottleneck_sizes:
        print(f"\n{'='*60}")
        print(f"BOTTLENECK: {bottleneck_dim} (threshold={input_dim})")
        print("=" * 60)
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = ThreeFactorConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                n_factor3=n_factor3,
                holdout_fraction=0.25,
                seed=seed
            )
            dataset = ThreeFactorDataset(config)
            
            model = ThreeFactorCompositionalModel(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                n_factor3=n_factor3
            )
            
            print(f"  Seed {seed+1}/{n_seeds}...", end=" ")
            history = train_three_factor_model(
                model, dataset, n_epochs=n_epochs, device=device
            )
            
            result = {
                "seed": seed,
                "train_acc_all": history["train_acc_all"][-1],
                "test_acc_all": history["test_acc_all"][-1],
                "test_acc_f1": history["test_acc_f1"][-1],
                "test_acc_f2": history["test_acc_f2"][-1],
                "test_acc_f3": history["test_acc_f3"][-1],
            }
            seed_results.append(result)
            
            print(f"Train: {result['train_acc_all']:.3f} | "
                  f"Test(all): {result['test_acc_all']:.3f} | "
                  f"F1: {result['test_acc_f1']:.3f} | "
                  f"F2: {result['test_acc_f2']:.3f} | "
                  f"F3: {result['test_acc_f3']:.3f}")
        
        # Aggregate
        test_all = [r["test_acc_all"] for r in seed_results]
        
        aggregate = {
            "bottleneck_dim": bottleneck_dim,
            "test_acc_all_mean": float(np.mean(test_all)),
            "test_acc_all_std": float(np.std(test_all)),
            "test_acc_f1_mean": float(np.mean([r["test_acc_f1"] for r in seed_results])),
            "test_acc_f2_mean": float(np.mean([r["test_acc_f2"] for r in seed_results])),
            "test_acc_f3_mean": float(np.mean([r["test_acc_f3"] for r in seed_results])),
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"\n  AGGREGATE: Test(all 3) = {aggregate['test_acc_all_mean']:.3f} "
              f"± {aggregate['test_acc_all_std']:.3f}")
    
    save_results(all_results, save_dir, "exp6_three_factors")
    
    return all_results


if __name__ == "__main__":
    # Bottleneck sweep for 3-factor experiment
    # Theoretical threshold: 8 + 8 + 4 = 20
    bottleneck_sizes = [10, 15, 20, 25, 30, 40, 50]
    
    results = run_three_factor_experiment(
        bottleneck_sizes=bottleneck_sizes,
        n_factor1=8,
        n_factor2=8,
        n_factor3=4,
        n_seeds=5,
        n_epochs=3000
    )
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    If 3-factor shows similar pattern to 2-factor:
    → Compositional output scales to higher dimensions
    → Strong support for end-to-end compositionality principle
    
    If 3-factor shows degraded performance:
    → Compositional generalization may not scale linearly
    → May need stronger architectural biases for complex compositions
    
    Key comparison:
    - 2-factor (Exp 4): 80% at threshold
    - 3-factor (Exp 6): ??? at threshold
    """)