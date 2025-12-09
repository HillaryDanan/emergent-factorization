"""
EXPERIMENT 2: Multi-Task Pressure

Tests whether multi-task demands force factorization when capacity pressure alone
might not.

HYPOTHESIS (APH Section 9.2, Condition 4 - Multi-Task Learning):
- A holistic encoder CANNOT solve all three tasks:
  * Task 1: Classify factor 1 only (must ignore factor 2)
  * Task 2: Classify factor 2 only (must ignore factor 1)
  * Task 3: Identify full combination
- Multi-task pressure should FORCE factorized representations
- Compositional generalization emerges as a SIDE EFFECT

KEY INSIGHT:
- In single-task (reconstruction), factorization is EFFICIENT but not NECESSARY
- In multi-task, factorization becomes NECESSARY to solve all tasks
- This is a STRONGER inductive bias than capacity pressure alone

PREDICTIONS:
1. Multi-task model should show higher compositional generalization than
   single-task model with equivalent bottleneck
2. Factor decodability should be higher in multi-task model
3. Effect should be robust across bottleneck sizes

CONTROLS:
1. Compare multi-task vs single-task with SAME architecture
2. Vary bottleneck size to see interaction effects
3. Ablate tasks (remove Task 3) to isolate factor-classification pressure

FALSIFICATION:
- If multi-task provides NO advantage → multi-task pressure unnecessary
- If multi-task helps but doesn't guarantee factorization → necessary but not sufficient

References:
- Multi-task learning and shared representations: Caruana (1997)
- Neural module networks (compositional pressure): Andreas et al. (2016)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .data import FactorialDataset, DatasetConfig
from .models import BottleneckAutoencoder, MultiTaskEncoder
from .analysis import analyze_representations, statistical_comparison
from .utils import set_seeds, get_device, save_results


def train_multitask_model(
    model: MultiTaskEncoder,
    dataset: FactorialDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cpu",
    verbose: bool = True
) -> Dict:
    """
    Train multi-task model with weighted loss.
    
    task_weights: (weight_factor1, weight_factor2, weight_combination)
    """
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [], "train_acc_f1": [], "train_acc_f2": [],
        "train_acc_comb": [], "test_acc_comb": [], "epoch": []
    }
    
    iterator = range(n_epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Multi-task training")
    
    for epoch in iterator:
        model.train()
        
        inputs, targets_comb, targets_f1, targets_f2 = dataset.get_batch("train")
        inputs = inputs.to(device)
        targets_comb = targets_comb.to(device)
        targets_f1 = targets_f1.to(device)
        targets_f2 = targets_f2.to(device)
        
        optimizer.zero_grad()
        logits_f1, logits_f2, logits_comb, _ = model(inputs)
        
        # Weighted multi-task loss
        loss_f1 = criterion(logits_f1, targets_f1)
        loss_f2 = criterion(logits_f2, targets_f2)
        loss_comb = criterion(logits_comb, targets_comb)
        
        loss = (
            task_weights[0] * loss_f1 +
            task_weights[1] * loss_f2 +
            task_weights[2] * loss_comb
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Train accuracies
                acc_f1 = (logits_f1.argmax(1) == targets_f1).float().mean().item()
                acc_f2 = (logits_f2.argmax(1) == targets_f2).float().mean().item()
                acc_comb = (logits_comb.argmax(1) == targets_comb).float().mean().item()
                
                # Test accuracy (compositional generalization)
                test_in, test_comb, test_f1, test_f2 = dataset.get_batch("test")
                test_in = test_in.to(device)
                test_comb = test_comb.to(device)
                _, _, test_logits, _ = model(test_in)
                test_acc = (test_logits.argmax(1) == test_comb).float().mean().item()
            
            history["train_loss"].append(loss.item())
            history["train_acc_f1"].append(acc_f1)
            history["train_acc_f2"].append(acc_f2)
            history["train_acc_comb"].append(acc_comb)
            history["test_acc_comb"].append(test_acc)
            history["epoch"].append(epoch)
            
            if verbose:
                iterator.set_postfix({
                    "F1": f"{acc_f1:.2f}",
                    "F2": f"{acc_f2:.2f}",
                    "Comb": f"{acc_comb:.2f}",
                    "Test": f"{test_acc:.2f}"
                })
    
    return history


def analyze_multitask_representations(
    model: MultiTaskEncoder,
    dataset: FactorialDataset,
    device: str = "cpu"
) -> Dict:
    """Analyze representations from multi-task model."""
    
    model.eval()
    inputs, _, f1_targets, f2_targets = dataset.get_batch("train")
    inputs = inputs.to(device)
    
    with torch.no_grad():
        _, _, _, representations = model(inputs)
        representations = representations.cpu().numpy()
    
    f1_targets = f1_targets.numpy()
    f2_targets = f2_targets.numpy()
    
    # Reuse analysis from single-task (same metrics apply)
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


def run_multitask_experiment(
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
    Compare single-task vs multi-task learning.
    
    For each bottleneck size:
    1. Train single-task autoencoder (baseline)
    2. Train multi-task encoder (with factor classification heads)
    3. Compare compositional generalization
    """
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT 2: MULTI-TASK PRESSURE")
    print("=" * 70)
    print(f"Comparing single-task vs multi-task across bottleneck sizes")
    print(f"Bottleneck sizes: {bottleneck_sizes}")
    print("=" * 70)
    
    all_results = {
        "experiment": "multitask_pressure",
        "config": {
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds,
            "n_epochs": n_epochs
        },
        "results": []
    }
    
    for bottleneck_dim in bottleneck_sizes:
        print(f"\n{'='*60}")
        print(f"BOTTLENECK: {bottleneck_dim}")
        print("=" * 60)
        
        single_task_results = []
        multi_task_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = DatasetConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=holdout_fraction,
                holdout_strategy="compositional",
                seed=seed
            )
            dataset = FactorialDataset(config)
            
            # ===== SINGLE-TASK (baseline) =====
            single_model = BottleneckAutoencoder(
                input_dim=dataset.n_combinations,
                bottleneck_dim=bottleneck_dim
            )
            
            print(f"  Seed {seed+1} - Single-task...", end=" ")
            from .experiment_bottleneck import train_model
            single_history = train_model(
                single_model, dataset, n_epochs=n_epochs,
                device=device, verbose=False
            )
            single_repr = analyze_representations(single_model, dataset, device)
            
            single_task_results.append({
                "seed": seed,
                "test_acc": single_history["test_acc"][-1],
                "train_acc": single_history["train_acc"][-1],
                "factor1_decode": single_repr["factor1_decodability"],
                "factor2_decode": single_repr["factor2_decodability"]
            })
            print(f"Test: {single_history['test_acc'][-1]:.3f}", end=" | ")
            
            # ===== MULTI-TASK =====
            set_seeds(seed)  # Reset for fair comparison
            
            multi_model = MultiTaskEncoder(
                input_dim=dataset.n_combinations,
                bottleneck_dim=bottleneck_dim,
                n_factor1=n_factor1,
                n_factor2=n_factor2
            )
            
            print(f"Multi-task...", end=" ")
            multi_history = train_multitask_model(
                multi_model, dataset, n_epochs=n_epochs,
                device=device, verbose=False
            )
            multi_repr = analyze_multitask_representations(multi_model, dataset, device)
            
            multi_task_results.append({
                "seed": seed,
                "test_acc": multi_history["test_acc_comb"][-1],
                "train_acc_f1": multi_history["train_acc_f1"][-1],
                "train_acc_f2": multi_history["train_acc_f2"][-1],
                "train_acc_comb": multi_history["train_acc_comb"][-1],
                "factor1_decode": multi_repr["factor1_decodability"],
                "factor2_decode": multi_repr["factor2_decodability"]
            })
            print(f"Test: {multi_history['test_acc_comb'][-1]:.3f}")
        
        # Statistical comparison
        single_test = [r["test_acc"] for r in single_task_results]
        multi_test = [r["test_acc"] for r in multi_task_results]
        
        comparison = statistical_comparison(
            multi_test, single_test,
            "multi_task", "single_task"
        )
        
        result = {
            "bottleneck_dim": bottleneck_dim,
            "single_task": {
                "test_acc_mean": np.mean(single_test),
                "test_acc_std": np.std(single_test),
                "seed_results": single_task_results
            },
            "multi_task": {
                "test_acc_mean": np.mean(multi_test),
                "test_acc_std": np.std(multi_test),
                "seed_results": multi_task_results
            },
            "comparison": comparison
        }
        all_results["results"].append(result)
        
        print(f"\n  COMPARISON:")
        print(f"    Single-task: {result['single_task']['test_acc_mean']:.3f} "
              f"± {result['single_task']['test_acc_std']:.3f}")
        print(f"    Multi-task:  {result['multi_task']['test_acc_mean']:.3f} "
              f"± {result['multi_task']['test_acc_std']:.3f}")
        print(f"    Cohen's d: {comparison['cohens_d']:.3f} "
              f"(p={comparison['p_value']:.4f})")
    
    save_results(all_results, save_dir, "exp2_multitask")
    
    return all_results


def run_task_ablation(
    bottleneck_dim: int = 20,
    n_factor1: int = 10,
    n_factor2: int = 10,
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Ablation study: Which tasks drive factorization?
    
    Conditions:
    1. Full multi-task (F1 + F2 + Combination)
    2. Factor classification only (F1 + F2, no combination)
    3. Single factor + combination (F1 + Combination)
    4. Single-task baseline (Combination only)
    
    If factor classification is the key driver, conditions 1 and 2
    should both outperform conditions 3 and 4.
    """
    
    if device is None:
        device = get_device()
    
    print("=" * 70)
    print("EXPERIMENT 2b: TASK ABLATION")
    print("=" * 70)
    
    conditions = {
        "full_multitask": (1.0, 1.0, 1.0),
        "factors_only": (1.0, 1.0, 0.0),
        "f1_and_comb": (1.0, 0.0, 1.0),
        "comb_only": (0.0, 0.0, 1.0)
    }
    
    all_results = {
        "experiment": "task_ablation",
        "config": {
            "bottleneck_dim": bottleneck_dim,
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "conditions": list(conditions.keys())
        },
        "results": {}
    }
    
    for condition_name, weights in conditions.items():
        print(f"\n--- {condition_name} (weights: {weights}) ---")
        
        condition_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = DatasetConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=0.25,
                holdout_strategy="compositional",
                seed=seed
            )
            dataset = FactorialDataset(config)
            
            model = MultiTaskEncoder(
                input_dim=dataset.n_combinations,
                bottleneck_dim=bottleneck_dim,
                n_factor1=n_factor1,
                n_factor2=n_factor2
            )
            
            history = train_multitask_model(
                model, dataset, n_epochs=n_epochs,
                task_weights=weights, device=device, verbose=False
            )
            
            repr_analysis = analyze_multitask_representations(model, dataset, device)
            
            condition_results.append({
                "seed": seed,
                "test_acc": history["test_acc_comb"][-1],
                "factor1_decode": repr_analysis["factor1_decodability"],
                "factor2_decode": repr_analysis["factor2_decodability"]
            })
        
        test_accs = [r["test_acc"] for r in condition_results]
        f1_decode = [r["factor1_decode"] for r in condition_results]
        f2_decode = [r["factor2_decode"] for r in condition_results]
        
        all_results["results"][condition_name] = {
            "test_acc_mean": np.mean(test_accs),
            "test_acc_std": np.std(test_accs),
            "factor1_decode_mean": np.mean(f1_decode),
            "factor2_decode_mean": np.mean(f2_decode),
            "seed_results": condition_results
        }
        
        print(f"  Test Acc: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
        print(f"  F1 Decode: {np.mean(f1_decode):.3f} | F2 Decode: {np.mean(f2_decode):.3f}")
    
    save_results(all_results, save_dir, "exp2b_ablation")
    
    return all_results


if __name__ == "__main__":
    # Run main multi-task experiment
    bottleneck_sizes = [10, 20, 30, 50, 100]
    
    results = run_multitask_experiment(
        bottleneck_sizes=bottleneck_sizes,
        n_seeds=5,
        n_epochs=2000
    )
    
    # Run ablation
    ablation_results = run_task_ablation(
        bottleneck_dim=20,
        n_seeds=5,
        n_epochs=2000
    )