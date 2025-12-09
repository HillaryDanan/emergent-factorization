"""
EXPERIMENT 3: Input Structure Effects

Tests whether RECOMBINATION EXPOSURE in input representation enables
factorization and compositional generalization.

HYPOTHESIS (APH Section 9.2, Condition 2 - Recombination Exposure):
- Factorized inputs provide recombination signal (same factor → same dimensions)
- This enables the network to DISCOVER that factors are separable
- Combined with bottleneck pressure, should yield factorized representations

DESIGN:
- Compare HOLISTIC inputs (one-hot over combinations) vs
- FACTORIZED inputs (concatenated factor one-hots)
- Same architecture, same bottleneck, same training

PREDICTIONS:
1. Holistic inputs → near-zero compositional generalization (as observed!)
2. Factorized inputs → high compositional generalization
3. Effect should be robust across bottleneck sizes

This directly tests whether Experiment 1's null result was due to
missing recombination exposure (as hypothesized) or something else.

FALSIFICATION:
- If factorized inputs ALSO show zero generalization → recombination exposure
  is insufficient; need other conditions or hypothesis is wrong
- If factorized inputs show high generalization → supports APH Section 9.2

References:
- Compositional generalization: Lake & Baroni (2018), ICML
- Importance of input structure: Montero et al. (2021), ICLR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from .data import FactorialDataset, DatasetConfig
from .data_factorized import FactorizedInputDataset, FactorizedDataConfig
from .models import BottleneckAutoencoder
from .analysis import analyze_representations, statistical_comparison
from .utils import set_seeds, get_device, save_results


def train_classifier(
    model: nn.Module,
    dataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    device: str = "cpu",
    verbose: bool = False
) -> Dict:
    """
    Train model for combination classification.
    
    Works with both holistic and factorized datasets.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch": []}
    
    for epoch in range(n_epochs):
        model.train()
        
        inputs, targets, _, _ = dataset.get_batch("train")
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(1) == targets).float().mean().item()
                
                test_in, test_tgt, _, _ = dataset.get_batch("test")
                test_in, test_tgt = test_in.to(device), test_tgt.to(device)
                test_logits, _ = model(test_in)
                test_acc = (test_logits.argmax(1) == test_tgt).float().mean().item()
            
            history["train_loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["epoch"].append(epoch)
    
    return history


def analyze_factorized_representations(
    model: nn.Module,
    dataset,
    device: str = "cpu"
) -> Dict:
    """Analyze representations (works for both dataset types)."""
    model.eval()
    
    inputs, _, f1_targets, f2_targets = dataset.get_batch("train")
    inputs = inputs.to(device)
    
    with torch.no_grad():
        _, representations = model(inputs)
        representations = representations.cpu().numpy()
    
    f1_targets = f1_targets.numpy()
    f2_targets = f2_targets.numpy()
    
    # Linear decodability
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import warnings
    
    results = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        clf1 = LogisticRegression(max_iter=1000, random_state=42)
        f1_scores = cross_val_score(clf1, representations, f1_targets, cv=5)
        results["factor1_decodability"] = float(np.mean(f1_scores))
        
        clf2 = LogisticRegression(max_iter=1000, random_state=42)
        f2_scores = cross_val_score(clf2, representations, f2_targets, cv=5)
        results["factor2_decodability"] = float(np.mean(f2_scores))
    
    return results


def run_input_structure_experiment(
    bottleneck_sizes: List[int],
    n_factor1: int = 10,
    n_factor2: int = 10,
    n_seeds: int = 5,
    n_epochs: int = 2000,
    holdout_fraction: float = 0.25,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Compare holistic vs factorized input representations.
    
    This is the KEY experiment: does input structure matter?
    """
    
    if device is None:
        device = get_device()
    
    n_combinations = n_factor1 * n_factor2
    factorized_input_dim = n_factor1 + n_factor2
    
    print("=" * 70)
    print("EXPERIMENT 3: INPUT STRUCTURE (Holistic vs Factorized)")
    print("=" * 70)
    print(f"Holistic input dim: {n_combinations}")
    print(f"Factorized input dim: {factorized_input_dim}")
    print(f"Bottleneck sizes: {bottleneck_sizes}")
    print("=" * 70)
    
    all_results = {
        "experiment": "input_structure",
        "config": {
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds,
            "holistic_input_dim": n_combinations,
            "factorized_input_dim": factorized_input_dim
        },
        "results": []
    }
    
    for bottleneck_dim in bottleneck_sizes:
        print(f"\n{'='*60}")
        print(f"BOTTLENECK: {bottleneck_dim}")
        print("=" * 60)
        
        holistic_results = []
        factorized_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            # ===== HOLISTIC INPUT (baseline - we know this fails) =====
            holistic_config = DatasetConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=holdout_fraction,
                holdout_strategy="compositional",
                seed=seed
            )
            holistic_data = FactorialDataset(holistic_config)
            
            holistic_model = BottleneckAutoencoder(
                input_dim=n_combinations,  # One-hot over all combinations
                bottleneck_dim=bottleneck_dim,
                hidden_dim=128
            )
            
            print(f"  Seed {seed+1} - Holistic...", end=" ")
            holistic_history = train_classifier(
                holistic_model, holistic_data, n_epochs=n_epochs,
                device=device, verbose=False
            )
            holistic_repr = analyze_factorized_representations(
                holistic_model, holistic_data, device
            )
            
            holistic_results.append({
                "seed": seed,
                "test_acc": holistic_history["test_acc"][-1],
                "train_acc": holistic_history["train_acc"][-1],
                "f1_decode": holistic_repr["factor1_decodability"],
                "f2_decode": holistic_repr["factor2_decodability"]
            })
            print(f"Test: {holistic_history['test_acc'][-1]:.3f}", end=" | ")
            
            # ===== FACTORIZED INPUT =====
            set_seeds(seed)  # Reset for fair comparison
            
            factorized_config = FactorizedDataConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=holdout_fraction,
                holdout_strategy="compositional",
                seed=seed
            )
            factorized_data = FactorizedInputDataset(factorized_config)
            
            factorized_model = BottleneckAutoencoder(
                input_dim=factorized_input_dim,  # Concatenated factor one-hots
                bottleneck_dim=bottleneck_dim,
                hidden_dim=128
            )
            
            print(f"Factorized...", end=" ")
            factorized_history = train_classifier(
                factorized_model, factorized_data, n_epochs=n_epochs,
                device=device, verbose=False
            )
            factorized_repr = analyze_factorized_representations(
                factorized_model, factorized_data, device
            )
            
            factorized_results.append({
                "seed": seed,
                "test_acc": factorized_history["test_acc"][-1],
                "train_acc": factorized_history["train_acc"][-1],
                "f1_decode": factorized_repr["factor1_decodability"],
                "f2_decode": factorized_repr["factor2_decodability"]
            })
            print(f"Test: {factorized_history['test_acc'][-1]:.3f}")
        
        # Statistical comparison
        holistic_test = [r["test_acc"] for r in holistic_results]
        factorized_test = [r["test_acc"] for r in factorized_results]
        
        comparison = statistical_comparison(
            factorized_test, holistic_test,
            "factorized", "holistic"
        )
        
        result = {
            "bottleneck_dim": bottleneck_dim,
            "holistic": {
                "test_acc_mean": float(np.mean(holistic_test)),
                "test_acc_std": float(np.std(holistic_test)),
                "f1_decode_mean": float(np.mean([r["f1_decode"] for r in holistic_results])),
                "f2_decode_mean": float(np.mean([r["f2_decode"] for r in holistic_results])),
                "seed_results": holistic_results
            },
            "factorized": {
                "test_acc_mean": float(np.mean(factorized_test)),
                "test_acc_std": float(np.std(factorized_test)),
                "f1_decode_mean": float(np.mean([r["f1_decode"] for r in factorized_results])),
                "f2_decode_mean": float(np.mean([r["f2_decode"] for r in factorized_results])),
                "seed_results": factorized_results
            },
            "comparison": comparison
        }
        all_results["results"].append(result)
        
        print(f"\n  COMPARISON:")
        print(f"    Holistic:   Test = {result['holistic']['test_acc_mean']:.3f} "
              f"± {result['holistic']['test_acc_std']:.3f} | "
              f"F1: {result['holistic']['f1_decode_mean']:.3f} | "
              f"F2: {result['holistic']['f2_decode_mean']:.3f}")
        print(f"    Factorized: Test = {result['factorized']['test_acc_mean']:.3f} "
              f"± {result['factorized']['test_acc_std']:.3f} | "
              f"F1: {result['factorized']['f1_decode_mean']:.3f} | "
              f"F2: {result['factorized']['f2_decode_mean']:.3f}")
        print(f"    Effect: Cohen's d = {comparison['cohens_d']:.2f}, "
              f"p = {comparison['p_value']:.4f}")
    
    save_results(all_results, save_dir, "exp3_input_structure")
    
    return all_results


if __name__ == "__main__":
    # Test across bottleneck sizes
    bottleneck_sizes = [5, 10, 15, 20, 25, 30, 50, 100]
    
    results = run_input_structure_experiment(
        bottleneck_sizes=bottleneck_sizes,
        n_seeds=5,
        n_epochs=2000
    )
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    If factorized inputs show HIGH compositional generalization:
    → Recombination exposure is necessary for factorization to emerge
    → Experiment 1's null result was due to missing this condition
    → Supports APH Section 9.2 framework
    
    If factorized inputs ALSO show LOW generalization:
    → Recombination exposure is insufficient
    → Need to investigate other conditions
    → May need to revise hypothesis
    """)