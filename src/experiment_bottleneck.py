"""
EXPERIMENT 1: Bottleneck Capacity Pressure

Tests whether bottleneck size determines emergence of factorized representations.

HYPOTHESIS (APH Section 9.2, Condition 1 - Factorization Pressure):
- Capacity constraints create pressure to discover efficient encodings
- For factorial data, factorization is the efficient encoding
- Medium bottleneck (≈ sum of factors) should yield factorization
- Large bottleneck (≈ product of factors) allows memorization

PREDICTIONS:
1. Medium bottleneck → high compositional generalization
2. Large bottleneck → low compositional generalization (memorization)
3. Possible phase transition around sum-of-factors threshold

FALSIFICATION:
- If compositional generalization is HIGH regardless of bottleneck size
  → Factorization pressure is unnecessary (hypothesis wrong)
- If compositional generalization is LOW regardless of bottleneck size
  → Capacity pressure alone is insufficient (need other conditions)
- If transition is gradual, not sharp
  → Compositionality may be graded, not binary (refines hypothesis)

References:
- Information bottleneck: Tishby et al. (2000)
- Compositional generalization failures: Lake & Baroni (2018)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from .data import FactorialDataset, DatasetConfig
from .models import BottleneckAutoencoder
from .analysis import analyze_representations, compute_compositionality_score
from .utils import set_seeds, get_device, save_results


def train_model(
    model: BottleneckAutoencoder,
    dataset: FactorialDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    device: str = "cpu",
    verbose: bool = True
) -> Dict:
    """Train autoencoder and track metrics."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch": []}
    
    iterator = range(n_epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training")
    
    for epoch in iterator:
        model.train()
        
        inputs, targets, _, _ = dataset.get_batch("train")
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        # Evaluate periodically
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_logits, _ = model(inputs)
                train_acc = (train_logits.argmax(1) == targets).float().mean().item()
                
                # Test accuracy (compositional generalization!)
                test_inputs, test_targets, _, _ = dataset.get_batch("test")
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_logits, _ = model(test_inputs)
                test_acc = (test_logits.argmax(1) == test_targets).float().mean().item()
            
            history["train_loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["epoch"].append(epoch)
            
            if verbose:
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "train": f"{train_acc:.3f}",
                    "test": f"{test_acc:.3f}"
                })
    
    return history


def run_bottleneck_experiment(
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
    Main bottleneck experiment.
    
    Runs across multiple bottleneck sizes and seeds.
    """
    
    if device is None:
        device = get_device()
    
    # Theoretical thresholds
    sum_factors = n_factor1 + n_factor2
    product_factors = n_factor1 * n_factor2
    
    print("=" * 70)
    print("EXPERIMENT 1: BOTTLENECK CAPACITY PRESSURE")
    print("=" * 70)
    print(f"Factorial structure: {n_factor1} × {n_factor2} = {product_factors}")
    print(f"Factorized capacity: {sum_factors} dimensions")
    print(f"Bottleneck sizes: {bottleneck_sizes}")
    print(f"Seeds per condition: {n_seeds}")
    print("=" * 70)
    
    all_results = {
        "experiment": "bottleneck_pressure",
        "config": {
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds,
            "n_epochs": n_epochs,
            "holdout_fraction": holdout_fraction,
            "sum_factors": sum_factors,
            "product_factors": product_factors
        },
        "results": []
    }
    
    for bottleneck_dim in bottleneck_sizes:
        print(f"\n{'='*60}")
        print(f"BOTTLENECK: {bottleneck_dim} "
              f"({bottleneck_dim/sum_factors:.2f}x sum, "
              f"{bottleneck_dim/product_factors:.2f}x product)")
        print("=" * 60)
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            # Create dataset
            config = DatasetConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=holdout_fraction,
                holdout_strategy="compositional",
                seed=seed
            )
            dataset = FactorialDataset(config)
            
            # Create model
            model = BottleneckAutoencoder(
                input_dim=dataset.n_combinations,
                bottleneck_dim=bottleneck_dim
            )
            
            # Train
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ")
            history = train_model(
                model, dataset, n_epochs=n_epochs,
                device=device, verbose=False
            )
            
            # Analyze
            repr_analysis = analyze_representations(model, dataset, device)
            
            result = {
                "seed": seed,
                "bottleneck_dim": bottleneck_dim,
                "final_train_acc": history["train_acc"][-1],
                "final_test_acc": history["test_acc"][-1],
                "representation_analysis": repr_analysis
            }
            seed_results.append(result)
            
            print(f"Train: {result['final_train_acc']:.3f} | "
                  f"Test: {result['final_test_acc']:.3f} | "
                  f"F1: {repr_analysis['factor1_decodability']:.3f} | "
                  f"F2: {repr_analysis['factor2_decodability']:.3f}")
        
        # Aggregate
        test_accs = [r["final_test_acc"] for r in seed_results]
        train_accs = [r["final_train_acc"] for r in seed_results]
        
        aggregate = {
            "bottleneck_dim": bottleneck_dim,
            "test_acc_mean": np.mean(test_accs),
            "test_acc_std": np.std(test_accs),
            "train_acc_mean": np.mean(train_accs),
            "train_acc_std": np.std(train_accs),
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"  → AGGREGATE: Test = {aggregate['test_acc_mean']:.3f} "
              f"± {aggregate['test_acc_std']:.3f}")
    
    # Save
    save_results(all_results, save_dir, "exp1_bottleneck")
    
    return all_results


if __name__ == "__main__":
    # Default experiment configuration
    bottleneck_sizes = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150]
    
    results = run_bottleneck_experiment(
        bottleneck_sizes=bottleneck_sizes,
        n_factor1=10,
        n_factor2=10,
        n_seeds=5,
        n_epochs=2000
    )