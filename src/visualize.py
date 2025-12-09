"""
Visualization for Emergent Factorization Results

Creates publication-ready figures summarizing experimental findings.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from typing import Dict, List, Optional


def set_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'sans-serif',
    })


def create_main_figure(save_dir: str = "results") -> plt.Figure:
    """
    Create the main results figure showing the key finding:
    Output structure is critical for compositional generalization.
    
    Layout:
    - Panel A: Comparison bar chart (non-comp vs comp output)
    - Panel B: Bottleneck sweep with compositional output
    - Panel C: Summary table as text
    """
    
    set_style()
    
    fig = plt.figure(figsize=(14, 5))
    
    # ===== PANEL A: Main Comparison =====
    ax1 = fig.add_subplot(131)
    
    conditions = ['Holistic Input\nNon-Comp Output', 
                  'Factorized Input\nNon-Comp Output',
                  'Factorized Input\nComp Output']
    means = [0.0, 0.0, 0.80]
    stds = [0.0, 0.0, 0.067]
    colors = ['#d62728', '#d62728', '#2ca02c']
    
    bars = ax1.bar(conditions, means, yerr=stds, capsize=5, color=colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax1.set_ylabel('Compositional Generalization\n(Both Factors Correct)')
    ax1.set_title('A. Output Structure is Critical', fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Chance (10%)')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.annotate(f'{mean:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.02),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Effect size annotation
    ax1.annotate('Cohen\'s d = 11.95', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== PANEL B: Bottleneck Sweep =====
    ax2 = fig.add_subplot(132)
    
    bottleneck_sizes = [5, 10, 15, 20, 25, 30, 50]
    test_means = [0.312, 0.512, 0.728, 0.800, 0.744, 0.672, 0.848]
    test_stds = [0.117, 0.082, 0.039, 0.067, 0.135, 0.157, 0.120]
    
    ax2.errorbar(bottleneck_sizes, test_means, yerr=test_stds, 
                 fmt='o-', capsize=5, color='#2ca02c', linewidth=2,
                 markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Mark theoretical threshold
    ax2.axvline(x=20, color='#1f77b4', linestyle='--', linewidth=2, alpha=0.7,
                label='Theoretical threshold\n(sum of factors = 20)')
    
    ax2.fill_between([0, 100], [0, 0], [0.1, 0.1], color='gray', alpha=0.2)
    ax2.text(27, 0.05, 'Chance', fontsize=9, color='gray')
    
    ax2.set_xlabel('Bottleneck Dimension')
    ax2.set_ylabel('Compositional Generalization')
    ax2.set_title('B. Bottleneck Sweep\n(Factorized Input + Compositional Output)', fontweight='bold')
    ax2.set_xlim([0, 55])
    ax2.set_ylim([0, 1.0])
    ax2.legend(loc='lower right', fontsize=9)
    
    # ===== PANEL C: Summary =====
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    summary_text = """
    KEY FINDING
    ══════════════════════════════════════
    
    Compositional generalization requires
    compositional structure END-TO-END:
    
    ✗ Holistic input + Non-comp output → 0%
    ✗ Factorized input + Non-comp output → 0%
    ✓ Factorized input + Comp output → 80%
    
    ══════════════════════════════════════
    
    INTERPRETATION
    
    Output structure acts as an INDUCTIVE
    BIAS that constrains the solution space.
    
    Non-compositional output permits
    memorization; compositional output
    FORCES compositional solutions.
    
    ══════════════════════════════════════
    
    THEORETICAL IMPORT
    
    Supports APH Section 9.2 with refinement:
    Conditions apply to ENTIRE pipeline,
    not just representation learning.
    
    Reference: Lake & Baroni (2018), ICML
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax3.set_title('C. Summary', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "main_results.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {fig_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(save_dir, "main_results.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    return fig


def create_2x2_matrix_figure(save_dir: str = "results") -> plt.Figure:
    """
    Create a 2x2 matrix showing the factorial design:
    
                    | Non-Comp Output | Comp Output
    ----------------+-----------------+------------
    Holistic Input  |      0%         |    ???
    Factorized Input|      0%         |    80%
    
    This clearly shows the experimental design and results.
    """
    
    set_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the matrix
    matrix_data = np.array([
        [0.0, np.nan],  # Holistic: non-comp=0%, comp=TBD
        [0.0, 0.80]     # Factorized: non-comp=0%, comp=80%
    ])
    
    # Color matrix
    colors = np.array([
        ['#d62728', '#cccccc'],  # Red, Gray (TBD)
        ['#d62728', '#2ca02c']   # Red, Green
    ])
    
    # Draw cells
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1-i), 1, 1, 
                                  facecolor=colors[i, j], 
                                  edgecolor='black', 
                                  linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            if np.isnan(matrix_data[i, j]):
                text = "TBD\n(Exp 5)"
                fontcolor = 'black'
            else:
                text = f"{matrix_data[i, j]:.0%}"
                fontcolor = 'white' if matrix_data[i, j] < 0.5 else 'white'
            
            ax.text(j + 0.5, 1.5 - i, text, 
                   ha='center', va='center', 
                   fontsize=24, fontweight='bold',
                   color=fontcolor)
    
    # Labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Non-Compositional\nOutput', 'Compositional\nOutput'], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Factorized\nInput', 'Holistic\nInput'], fontsize=12)
    
    ax.set_xlabel('Output Structure', fontsize=14, fontweight='bold')
    ax.set_ylabel('Input Structure', fontsize=14, fontweight='bold')
    ax.set_title('Factorial Design: Input × Output Structure\n' + 
                 'Values show Compositional Generalization Rate', 
                 fontsize=14, fontweight='bold')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(length=0)
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, "factorial_design.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {fig_path}")
    
    plt.show()
    
    return fig


def create_all_figures(save_dir: str = "results"):
    """Generate all figures."""
    print("Generating main results figure...")
    create_main_figure(save_dir)
    
    print("\nGenerating factorial design figure...")
    create_2x2_matrix_figure(save_dir)
    
    print("\nAll figures generated!")


if __name__ == "__main__":
    create_all_figures()