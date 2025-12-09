"""
Run complete experimental suite for emergent factorization.

EXPERIMENTS:
1-3: Baseline (null results with non-compositional output)
4: Compositional output (KEY FINDING: 80% generalization)
5: Full factorial (isolate input vs output contributions)
6: Three-factor extension (test scalability)
"""

import os
import sys
from datetime import datetime

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device


def main():
    print("=" * 70)
    print("EMERGENT FACTORIZATION: COMPLETE EXPERIMENTAL SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = get_device()
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # ===== EXPERIMENT 5: Full Factorial =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Full 2×2 Factorial Design")
    print("=" * 70)
    
    from src.experiment_holistic_compositional import run_full_factorial_comparison
    
    exp5_results = run_full_factorial_comparison(
        bottleneck_dim=20,
        n_seeds=10,
        n_epochs=2000,
        device=device,
        save_dir=save_dir
    )
    
    # ===== EXPERIMENT 6: Three Factors =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Three-Factor Extension")
    print("=" * 70)
    
    from src.experiment_three_factors import run_three_factor_experiment
    
    exp6_results = run_three_factor_experiment(
        bottleneck_sizes=[10, 15, 20, 25, 30, 40, 50],
        n_factor1=8,
        n_factor2=8,
        n_factor3=4,
        n_seeds=5,
        n_epochs=3000,
        device=device,
        save_dir=save_dir
    )
    
    # ===== GENERATE FIGURES =====
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    from src.visualize import create_all_figures
    create_all_figures(save_dir)
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("COMPLETE SUMMARY")
    print("=" * 70)
    
    print("\n2×2 Factorial Results (Experiment 5):")
    print("                    | Non-Comp     | Comp")
    print("  ------------------+--------------+--------------")
    if "summary" in exp5_results:
        A = exp5_results["summary"]["holistic_noncomp"]["mean"]
        B = exp5_results["summary"]["holistic_comp"]["mean"]
        C = exp5_results["summary"]["factorized_noncomp"]["mean"]
        D = exp5_results["summary"]["factorized_comp"]["mean"]
        print(f"  Holistic Input    | {A:.1%}         | {B:.1%}")
        print(f"  Factorized Input  | {C:.1%}         | {D:.1%}")
    
    print("\n3-Factor Results (Experiment 6):")
    for r in exp6_results["results"]:
        print(f"  Bottleneck {r['bottleneck_dim']:3d}: "
              f"{r['test_acc_all_mean']:.1%} ± {r['test_acc_all_std']:.1%}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()