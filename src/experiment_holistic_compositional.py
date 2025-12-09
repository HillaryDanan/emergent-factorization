"""
EXPERIMENT 5: Holistic Input + Compositional Output

MOTIVATION:
Experiment 4 showed that factorized input + compositional output → 80% generalization.
But we changed TWO things from the baseline (Experiments 1-3).

This experiment isolates the contribution of OUTPUT structure by testing:
- Holistic input (one-hot over combinations) + Compositional output (f1 + f2 heads)

DESIGN:
                    | Non-Comp Output | Comp Output
    ----------------+-----------------+------------
    Holistic Input  |      0% (Exp1)  |  ??? (Exp5)
    Factorized Input|      0% (Exp3)  |  80% (Exp4)

PREDICTIONS:

Hypothesis A (Output is everything):
- If output structure is the ONLY thing that matters
- Exp 5 should show ~80% (same as Exp 4)
- This would mean input structure doesn't matter

Hypothesis B (Both required):
- If BOTH factorized input AND compositional output are required
- Exp 5 should show ~0% or very low (similar to Exp 1-3)
- This would mean both conditions are necessary

Hypothesis C (Partial contribution):
- If input and output structure BOTH contribute
- Exp 5 should show intermediate performance (e.g., 20-50%)
- This would suggest both matter, with possible interaction

THEORETICAL IMPORT:
This determines whether the APH conditions are:
- Disjunctive (any one is sufficient)
- Conjunctive (all required)
- Additive (each contributes independently)
- Interactive (combination is greater than sum)

References:
- Factorial experimental design: Fisher (1935)
- Interaction effects in learning: Caruana (1997)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .data import FactorialDataset, DatasetConfig
from .utils import set_seeds, get_device, save_results


class HolisticCompositionalModel(nn.Module):
    """
    Model with:
    - HOLISTIC input (one-hot over combinations, 100-dim for 10x10)
    - COMPOSITIONAL output (separate f1 and f2 heads)
    
    This is the missing cell in our 2x2 factorial design.
    """
    
    def __init__(
        self,
        input_dim: int,  # n_combinations (e.g., 100)
        bottleneck_dim: int,
        n_factor1: int = 10,
        n_factor2: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Encoder (same as before)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        # COMPOSITIONAL output: separate heads for each factor
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
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits_f1 = self.head_f1(z)
        logits_f2 = self.head_f2(z)
        return logits_f1, logits_f2, z


def train_holistic_compositional(
    model: HolisticCompositionalModel,
    dataset: FactorialDataset,
    n_epochs: int = 2000,
    learning_rate: float = 0.001,
    device: str = "cpu",
    verbose: bool = False
) -> Dict:
    """Train model with holistic input, compositional output."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "train_acc_f1": [], "train_acc_f2": [], "train_acc_both": [],
        "test_acc_f1": [], "test_acc_f2": [], "test_acc_both": [],
        "epoch": []
    }
    
    for epoch in range(n_epochs):
        model.train()
        
        # Note: dataset.get_batch returns (holistic_input, combo_target, f1_target, f2_target)
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
                
                # Test metrics
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


def run_holistic_compositional_experiment(
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
    Test holistic input + compositional output.
    
    This completes the 2x2 factorial design.
    """
    
    if device is None:
        device = get_device()
    
    n_combinations = n_factor1 * n_factor2
    
    print("=" * 70)
    print("EXPERIMENT 5: HOLISTIC INPUT + COMPOSITIONAL OUTPUT")
    print("=" * 70)
    print(f"Input: HOLISTIC ({n_combinations} dims, one-hot over combinations)")
    print(f"Output: COMPOSITIONAL (f1 + f2 heads)")
    print(f"Test: BOTH factors correct on novel combinations")
    print("=" * 70)
    
    all_results = {
        "experiment": "holistic_compositional",
        "config": {
            "n_factor1": n_factor1,
            "n_factor2": n_factor2,
            "bottleneck_sizes": bottleneck_sizes,
            "n_seeds": n_seeds,
            "input_structure": "holistic",
            "output_structure": "compositional"
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
            
            config = DatasetConfig(
                n_factor1=n_factor1,
                n_factor2=n_factor2,
                holdout_fraction=holdout_fraction,
                holdout_strategy="compositional",
                seed=seed
            )
            dataset = FactorialDataset(config)
            
            model = HolisticCompositionalModel(
                input_dim=n_combinations,  # Holistic!
                bottleneck_dim=bottleneck_dim,
                n_factor1=n_factor1,
                n_factor2=n_factor2
            )
            
            print(f"  Seed {seed+1}/{n_seeds}...", end=" ")
            history = train_holistic_compositional(
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
                "test_acc_both": history["test_acc_both"][-1],
            }
            seed_results.append(result)
            
            print(f"Train: {result['train_acc_both']:.3f} | "
                  f"Test: {result['test_acc_both']:.3f} | "
                  f"F1: {result['test_acc_f1']:.3f} | "
                  f"F2: {result['test_acc_f2']:.3f}")
        
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
            "seed_results": seed_results
        }
        all_results["results"].append(aggregate)
        
        print(f"\n  AGGREGATE:")
        print(f"    Train (both): {aggregate['train_acc_both_mean']:.3f}")
        print(f"    Test (both):  {aggregate['test_acc_both_mean']:.3f} "
              f"± {aggregate['test_acc_both_std']:.3f}")
    
    save_results(all_results, save_dir, "exp5_holistic_compositional")
    
    return all_results


def run_full_factorial_comparison(
    bottleneck_dim: int = 20,
    n_factor1: int = 10,
    n_factor2: int = 10,
    n_seeds: int = 10,  # More seeds for precise comparison
    n_epochs: int = 2000,
    device: Optional[str] = None,
    save_dir: str = "results"
) -> Dict:
    """
    Run all 4 conditions of the 2x2 factorial design.
    
    This gives us the complete picture:
    
                        | Non-Comp Output | Comp Output
    --------------------+-----------------+------------
    Holistic Input      |   Condition A   | Condition B (Exp 5)
    Factorized Input    |   Condition C   | Condition D (Exp 4)
    """
    
    if device is None:
        device = get_device()
    
    from .data_factorized import FactorizedInputDataset, FactorizedDataConfig
    from .models import BottleneckAutoencoder
    from .experiment_compositional_output import CompositionalModel, train_compositional_model
    
    print("=" * 70)
    print("FULL 2×2 FACTORIAL COMPARISON")
    print("=" * 70)
    print(f"Bottleneck: {bottleneck_dim}")
    print(f"Seeds: {n_seeds}")
    print("=" * 70)
    
    n_combinations = n_factor1 * n_factor2
    factorized_dim = n_factor1 + n_factor2
    
    results = {
        "holistic_noncomp": [],      # Condition A
        "holistic_comp": [],          # Condition B
        "factorized_noncomp": [],     # Condition C
        "factorized_comp": []         # Condition D
    }
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")
        
        # ===== CONDITION A: Holistic + Non-Compositional =====
        set_seeds(seed)
        
        holistic_config = DatasetConfig(
            n_factor1=n_factor1, n_factor2=n_factor2,
            holdout_fraction=0.25, holdout_strategy="compositional", seed=seed
        )
        holistic_data = FactorialDataset(holistic_config)
        
        model_a = BottleneckAutoencoder(
            input_dim=n_combinations,
            bottleneck_dim=bottleneck_dim
        )
        model_a = model_a.to(device)
        optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(n_epochs):
            model_a.train()
            inputs, targets, _, _ = holistic_data.get_batch("train")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_a.zero_grad()
            logits, _ = model_a(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_a.step()
        
        model_a.eval()
        with torch.no_grad():
            test_in, test_tgt, _, _ = holistic_data.get_batch("test")
            test_in, test_tgt = test_in.to(device), test_tgt.to(device)
            test_logits, _ = model_a(test_in)
            acc_a = (test_logits.argmax(1) == test_tgt).float().mean().item()
        
        results["holistic_noncomp"].append(acc_a)
        print(f"  A (Holistic + Non-Comp):   {acc_a:.3f}")
        
        # ===== CONDITION B: Holistic + Compositional =====
        set_seeds(seed)
        
        model_b = HolisticCompositionalModel(
            input_dim=n_combinations,
            bottleneck_dim=bottleneck_dim,
            n_factor1=n_factor1, n_factor2=n_factor2
        )
        
        history_b = train_holistic_compositional(
            model_b, holistic_data, n_epochs=n_epochs,
            device=device, verbose=False
        )
        acc_b = history_b["test_acc_both"][-1]
        results["holistic_comp"].append(acc_b)
        print(f"  B (Holistic + Comp):       {acc_b:.3f}")
        
        # ===== CONDITION C: Factorized + Non-Compositional =====
        set_seeds(seed)
        
        factorized_config = FactorizedDataConfig(
            n_factor1=n_factor1, n_factor2=n_factor2,
            holdout_fraction=0.25, holdout_strategy="compositional", seed=seed
        )
        factorized_data = FactorizedInputDataset(factorized_config)
        
        model_c = BottleneckAutoencoder(
            input_dim=factorized_dim,
            bottleneck_dim=bottleneck_dim
        )
        # Adjust output dimension
        model_c.decoder[-1] = nn.Linear(128, n_combinations)
        model_c = model_c.to(device)
        optimizer_c = optim.Adam(model_c.parameters(), lr=0.001)
        
        for epoch in range(n_epochs):
            model_c.train()
            inputs, targets, _, _ = factorized_data.get_batch("train")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_c.zero_grad()
            logits, _ = model_c(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_c.step()
        
        model_c.eval()
        with torch.no_grad():
            test_in, test_tgt, _, _ = factorized_data.get_batch("test")
            test_in, test_tgt = test_in.to(device), test_tgt.to(device)
            test_logits, _ = model_c(test_in)
            acc_c = (test_logits.argmax(1) == test_tgt).float().mean().item()
        
        results["factorized_noncomp"].append(acc_c)
        print(f"  C (Factorized + Non-Comp): {acc_c:.3f}")
        
        # ===== CONDITION D: Factorized + Compositional =====
        set_seeds(seed)
        
        model_d = CompositionalModel(
            input_dim=factorized_dim,
            bottleneck_dim=bottleneck_dim,
            n_factor1=n_factor1, n_factor2=n_factor2
        )
        
        history_d = train_compositional_model(
            model_d, factorized_data, n_epochs=n_epochs,
            device=device, verbose=False
        )
        acc_d = history_d["test_acc_both"][-1]
        results["factorized_comp"].append(acc_d)
        print(f"  D (Factorized + Comp):     {acc_d:.3f}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("FULL FACTORIAL RESULTS")
    print("=" * 70)
    
    summary = {}
    for cond, accs in results.items():
        summary[cond] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "values": accs
        }
        print(f"  {cond:25s}: {summary[cond]['mean']:.3f} ± {summary[cond]['std']:.3f}")
    
    # Compute interaction effect
    # Interaction = (D - C) - (B - A) = D - C - B + A
    A = np.mean(results["holistic_noncomp"])
    B = np.mean(results["holistic_comp"])
    C = np.mean(results["factorized_noncomp"])
    D = np.mean(results["factorized_comp"])
    
    main_effect_input = ((C + D) - (A + B)) / 2
    main_effect_output = ((B + D) - (A + C)) / 2
    interaction = ((D - C) - (B - A))
    
    print(f"\n  Main effect of INPUT:  {main_effect_input:.3f}")
    print(f"  Main effect of OUTPUT: {main_effect_output:.3f}")
    print(f"  Interaction:           {interaction:.3f}")
    
    print("\n  2×2 Matrix:")
    print(f"                    | Non-Comp     | Comp")
    print(f"  ------------------+--------------+--------------")
    print(f"  Holistic Input    | {A:.1%}        | {B:.1%}")
    print(f"  Factorized Input  | {C:.1%}        | {D:.1%}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if B > 0.5 and D > 0.5:
        print("  → OUTPUT structure is sufficient for generalization")
        print("    (Both B and D show high performance)")
    elif D > 0.5 and B < 0.2:
        print("  → BOTH input and output structure required (conjunctive)")
        print("    (Only D shows high performance)")
    elif D > 0.5 and 0.2 < B < 0.5:
        print("  → Both contribute, with possible interaction")
        print("    (D > B, but B > A,C)")
    else:
        print("  → Unexpected result - investigate further")
    
    # Save
    full_results = {
        "experiment": "full_factorial",
        "config": {
            "bottleneck_dim": bottleneck_dim,
            "n_seeds": n_seeds
        },
        "summary": summary,
        "effects": {
            "main_effect_input": main_effect_input,
            "main_effect_output": main_effect_output,
            "interaction": interaction
        }
    }
    save_results(full_results, save_dir, "exp5_full_factorial")
    
    return full_results


if __name__ == "__main__":
    # Run complete factorial comparison
    results = run_full_factorial_comparison(
        bottleneck_dim=20,
        n_seeds=10,
        n_epochs=2000
    )