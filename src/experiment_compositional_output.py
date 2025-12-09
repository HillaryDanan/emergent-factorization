"""
EXPERIMENT 4: Compositional Output Structure

DIAGNOSIS FROM EXPERIMENT 3:
- Factorized inputs led to Factor 2 being preserved (~93% decodable)
- But test accuracy was 0%
- WHY? The output (100-way combination classification) is NON-COMPOSITIONAL

INSIGHT:
Compositional generalization requires compositional structure END-TO-END:
- Input structure: factorized ✓ (we fixed this)
- Output structure: must ALSO be factorized

THE FIX:
Instead of predicting combination index (non-compositional), predict:
- Factor 1 (10-way classification)
- Factor 2 (10-way classification)

Test metric: % of novel combinations where BOTH factors correct.

HYPOTHESIS:
With compositional input AND compositional output, the network should
achieve high compositional generalization (predicting both factors
correctly for novel combinations).

THEORETICAL IMPORT:
This refines APH Section 9.2: compositional abstraction requires
compositional structure throughout the processing pipeline, not just
in representation learning.

References:
- SCAN benchmark uses compositional output: Lake & Baroni (2018)
- Systematic generalization requires structural alignment: Fodor & Pylyshyn (1988)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .data_factorized import FactorizedInputDataset, FactorizedDataConfig
from .utils import set_seeds, get_device, save_results


class CompositionalModel(nn.Module):
    """
    Model with compositional output structure.
    
    Architecture:
        Factorized Input → Encoder → Bottleneck → Factor1 Head (10-way)
                                                → Factor2 Head (10-way)
    
    NO combination classification head. Output is purely factorized.
    
    This tests whether compositional input + compositional output
    enables compositional generalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        n_factor1: int = 10,
        n_factor2: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_factor1 = n_factor1
        self.n_factor2 = n_factor2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        # Factor 1 prediction head
        self.head_f1 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor1)
        )
        
        # Factor 2 prediction head
        self.head_f2 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits_f1: Factor 1 predictions
            logits_f2: Factor 2 predictions  
            bottleneck: Bottleneck representation
        """
        z = self.encode(x)
        logits_f1 = self.head_f1(z)
        logits_f2 = self.head_f2(z)
        return logits_f1, logits_f2, z


def train_compositional_model(
    model: CompositionalModel,
    dataset: FactorizedInputDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    device: str = "cpu",
    verbose: bool = False
) -> Dict:
    """
    Train model to predict both factors.
    
    Loss = CrossEntropy(f1_pred, f1_true) + CrossEntropy(f2_pred, f2_true)
    
    NO combination classification.
    """
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "train_acc_f1": [],
        "train_acc_f2": [],
        "train_acc_both": [],  # Both factors correct
        "test_acc_f1": [],
        "test_acc_f2": [],
        "test_acc_both": [],  # KEY METRIC: compositional generalization
        "epoch": []
    }
    
    for epoch in range(n_epochs):
        model.train()
        
        inputs, _, f1_targets, f2_targets = dataset.get_batch("train")
        inputs = inputs.to(device)
        f1_targets = f1_targets.to(device)
        f2_targets = f2_targets.to(device)
        
        optimizer.zero_grad()
        logits_f1, logits_f2, _ = model(inputs)
        
        loss_f1 = criterion(logits_f1, f1_targets)
        loss_f2 = criterion(logits_f2, f2_targets)
        loss = loss_f1 + loss_f2
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Training metrics
                pred_f1 = logits_f1.argmax(1)
                pred_f2 = logits_f2.argmax(1)
                
                acc_f1 = (pred_f1 == f1_targets).float().mean().item()
                acc_f2 = (pred_f2 == f2_targets).float().mean().item()
                acc_both = ((pred_f1 == f1_targets) & (pred_f2 == f2_targets)).float().mean().item()
                
                # Test metrics (COMPOSITIONAL GENERALIZATION)
                test_in, _, test_f1, test_f2 = dataset.get_batch("test")
                test_in = test_in.to(device)
                test_f1 = test_f1.to(device)
                test_f2 = test_f2.to(device)
                
                test_logits_f1, test_logits_f2, _ = model(test_in)
                test_pred_f1 = test_logits_f1.argmax(1)
                test_pred_f2 = test_logits_f2.argmax(1)
                
                test_acc_f1 = (test_pred_f1 == test_f1).float().mean().item()
                test_acc_f2 = (test_pred_f2 == test_f2).float().mean().item()
                test_acc_both = ((test_pred_f1 == test_f1) & (test_pred_f2 == test_f2)).float().mean().item()
            
            history["train_loss"].append(loss.item())
            history["train_acc_f1"].append(acc_f1)
            history["train_acc_f2"].append(acc_f2)
            history["train_acc_both"].append(acc_both)
            history["test_acc_f1"].append(test_acc_f1)
            history["test_acc_f2"].append(test_acc_f2)
            history["test_acc_both"].append(test_acc_both)
            history["epoch"].append(epoch)
    
    return history


def run_compositional_output_experiment(
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
    Test compositional generalization with compositional output structure.
    
    KEY CHANGE from previous experiments:
    - Output is (f1_prediction, f2_prediction), not combination_index
    - Test metric is: both factors correct for novel combinations
    """
    
    if device is None:
        device = get_device()
    
    input_dim = n_factor1 + n_factor2
    
    print("=" * 70)
    print("EXPERIMENT 4: COMPOSITIONAL OUTPUT STRUCTURE")
    print("=" * 70)
    print(f"Input: factorized ({input_dim} dims)")
    print(f"Output: factorized (predict f1 and f2 separately)")
    print(f"Test metric: BOTH factors correct on novel combinations")
    print(f"Bottleneck sizes: {bottleneck_sizes}")
    print("=" * 70)
    
    all_results = {
        "experiment": "compositional_output",
        "config": {
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds,
            "input_dim": input_dim,
            "output_structure": "factorized (f1 + f2 heads)"
        },
        "results": []
    }
    
    for bottleneck_dim in bottleneck_sizes:
        print(f"\n{'='*60}")
        print(f"BOTTLENECK: {bottleneck_dim}")
        print("=" * 60)
        
        seed_results = []
        
        for seed in range(n_seeds):
            set_seeds(seed)
            
            config = FactorizedDataConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=holdout_fraction,
                holdout_strategy="compositional",
                seed=seed
            )
            dataset = FactorizedInputDataset(config)
            
            model = CompositionalModel(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                n_factor1=n_factor1,
                n_factor2=n_factor2
            )
            
            print(f"  Seed {seed+1}/{n_seeds}...", end=" ")
            history = train_compositional_model(
                model, dataset, n_epochs=n_epochs,
                device=device, verbose=False
            )
            
            result = {
                "seed": seed,
                "train_acc_f1": history["train_acc_f1"][-1],
                "train_acc_f2": history["train_acc_f2"][-1],
                "train_acc_both": history["train_acc_both"][-1],
                "test_acc_f1": history["test_acc_f1"][-1],
                "test_acc_f2": history["test_acc_f2"][-1],
                "test_acc_both": history["test_acc_both"][-1],  # KEY METRIC
            }
            seed_results.append(result)
            
            print(f"Train(both): {result['train_acc_both']:.3f} | "
                  f"Test(both): {result['test_acc_both']:.3f} | "
                  f"Test(F1): {result['test_acc_f1']:.3f} | "
                  f"Test(F2): {result['test_acc_f2']:.3f}")
        
        # Aggregate
        test_both = [r["test_acc_both"] for r in seed_results]
        test_f1 = [r["test_acc_f1"] for r in seed_results]
        test_f2 = [r["test_acc_f2"] for r in seed_results]
        train_both = [r["train_acc_both"] for r in seed_results]
        
        aggregate = {
            "bottleneck_dim": bottleneck_dim,
            "test_acc_both_mean": float(np.mean(test_both)),
            "test_acc_both_std": float(np.std(test_both)),
            "test_acc_f1_mean": float(np.mean(test_f1)),
            "test_acc_f2_mean": float(np.mean(test_f2)),
            "train_acc_both_mean": float(np.mean(train_both)),
            "generalization_gap": float(np.mean(train_both) - np.mean(test_both)),
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"\n  AGGREGATE:")
        print(f"    Train (both correct): {aggregate['train_acc_both_mean']:.3f}")
        print(f"    Test (both correct):  {aggregate['test_acc_both_mean']:.3f} "
              f"± {aggregate['test_acc_both_std']:.3f}")
        print(f"    Generalization gap:   {aggregate['generalization_gap']:.3f}")
    
    save_results(all_results, save_dir, "exp4_compositional_output")
    
    return all_results


def run_comparison_with_original(
    n_seeds: int = 5,
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Direct comparison: Original architecture vs Compositional architecture
    
    This isolates the effect of output structure.
    Both use:
    - Factorized input (20 dims)
    - Bottleneck = 20 (at factorization threshold)
    
    Difference:
    - Original: 100-way combination classification (non-compositional output)
    - Compositional: 10-way f1 + 10-way f2 (compositional output)
    """
    
    if device is None:
        device = get_device()
    
    from .data_factorized import FactorizedInputDataset, FactorizedDataConfig
    from .models import BottleneckAutoencoder
    
    print("=" * 70)
    print("COMPARISON: Non-Compositional vs Compositional Output")
    print("=" * 70)
    print("Both use factorized input, bottleneck=20")
    print("Difference: output structure")
    print("=" * 70)
    
    n_factor1, n_factor2 = 10, 10
    bottleneck_dim = 20
    input_dim = n_factor1 + n_factor2
    
    results = {
        "non_compositional_output": [],
        "compositional_output": []
    }
    
    for seed in range(n_seeds):
        set_seeds(seed)
        
        config = FactorizedDataConfig(
            n_factor1=n_factor1,
            n_factor2=n_factor2,
            holdout_fraction=0.25,
            holdout_strategy="compositional",
            seed=seed
        )
        dataset = FactorizedInputDataset(config)
        
        # --- NON-COMPOSITIONAL OUTPUT (100-way classification) ---
        from .experiment_input_structure import train_classifier
        
        non_comp_model = BottleneckAutoencoder(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim
        )
        # Fix: output dim should match n_combinations
        non_comp_model.decoder[-1] = nn.Linear(128, n_factor1 * n_factor2)
        
        print(f"Seed {seed+1} - Non-compositional output...", end=" ")
        
        # Custom training for this model
        non_comp_model = non_comp_model.to(device)
        optimizer = optim.Adam(non_comp_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(n_epochs):
            non_comp_model.train()
            inputs, targets, _, _ = dataset.get_batch("train")
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = non_comp_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        non_comp_model.eval()
        with torch.no_grad():
            test_in, test_tgt, _, _ = dataset.get_batch("test")
            test_in, test_tgt = test_in.to(device), test_tgt.to(device)
            test_logits, _ = non_comp_model(test_in)
            non_comp_test_acc = (test_logits.argmax(1) == test_tgt).float().mean().item()
        
        results["non_compositional_output"].append(non_comp_test_acc)
        print(f"Test: {non_comp_test_acc:.3f}", end=" | ")
        
        # --- COMPOSITIONAL OUTPUT (f1 + f2 heads) ---
        set_seeds(seed)
        
        comp_model = CompositionalModel(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            n_factor1=n_factor1,
            n_factor2=n_factor2
        )
        
        print(f"Compositional output...", end=" ")
        history = train_compositional_model(
            comp_model, dataset, n_epochs=n_epochs,
            device=device, verbose=False
        )
        
        comp_test_acc = history["test_acc_both"][-1]
        results["compositional_output"].append(comp_test_acc)
        print(f"Test: {comp_test_acc:.3f}")
    
    # Summary
    non_comp_mean = np.mean(results["non_compositional_output"])
    non_comp_std = np.std(results["non_compositional_output"])
    comp_mean = np.mean(results["compositional_output"])
    comp_std = np.std(results["compositional_output"])
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Non-compositional output: {non_comp_mean:.3f} ± {non_comp_std:.3f}")
    print(f"  Compositional output:     {comp_mean:.3f} ± {comp_std:.3f}")
    print(f"  Effect size (Cohen's d):  {(comp_mean - non_comp_mean) / max(comp_std, 0.01):.2f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run the key comparison first
    print("\n" + "="*70)
    print("PART 1: Direct Comparison (same input, different output)")
    print("="*70 + "\n")
    
    comparison_results = run_comparison_with_original(
        n_seeds=5,
        n_epochs=2000
    )
    
    # Then sweep bottleneck sizes
    print("\n" + "="*70)
    print("PART 2: Bottleneck Sweep with Compositional Output")
    print("="*70 + "\n")
    
    sweep_results = run_compositional_output_experiment(
        bottleneck_sizes=[5, 10, 15, 20, 25, 30, 50],
        n_seeds=5,
        n_epochs=2000
    )
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    If compositional output shows HIGH generalization:
    → Output structure was the missing piece
    → Compositional abstraction requires end-to-end compositional structure
    → Refines APH: conditions apply to entire pipeline, not just encoding
    
    If compositional output ALSO shows LOW generalization:
    → Something else is wrong (architecture, optimization, task structure)
    → Need deeper investigation
    
    Expected result based on diagnosis:
    - Non-compositional output: ~0% (confirmed by Exp 3)
    - Compositional output: HIGH (prediction)
    """)