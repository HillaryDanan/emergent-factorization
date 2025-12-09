"""
Visualizations for Extended Experiments

Creates publication-ready figures for:
1. Scaling (N-factors)
2. Asymmetry effects
3. Noise robustness
4. Architecture comparison

Design principles:
- Clean, minimal aesthetic
- Consistent color scheme
- Clear labeling
- Publication-ready (300 DPI, vector-compatible)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from typing import Dict, List, Optional


def set_publication_style():
    """Set clean publication style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# Color scheme
COLORS = {
    'primary': '#2ca02c',      # Green - success/high performance
    'secondary': '#1f77b4',    # Blue - reference/threshold
    'warning': '#ff7f0e',      # Orange - moderate
    'danger': '#d62728',       # Red - failure/low performance
    'neutral': '#7f7f7f',      # Gray - baseline
    'highlight': '#9467bd',    # Purple - special
}


def create_scaling_figure(save_dir: str = "results") -> plt.Figure:
    """
    Figure 2A: Scaling with number of factors.
    
    Shows that compression ratio predicts performance.
    """
    set_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Data from experiment
    n_factors = [2, 3, 4, 5]
    combinations = [16, 64, 256, 1024]
    compression_ratios = [2.0, 5.3, 16.0, 51.2]
    test_accs = [0.20, 0.912, 1.0, 1.0]
    test_stds = [0.187, 0.064, 0.0, 0.0]
    
    # Panel 1: By number of factors
    colors = [COLORS['danger'], COLORS['warning'], COLORS['primary'], COLORS['primary']]
    bars = ax1.bar(n_factors, test_accs, yerr=test_stds, capsize=5, 
                   color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    
    ax1.set_xlabel('Number of Factors')
    ax1.set_ylabel('Compositional Generalization')
    ax1.set_title('A. Scaling with Factorial Complexity')
    ax1.set_ylim([0, 1.1])
    ax1.set_xticks(n_factors)
    
    # Add compression ratio labels
    for i, (bar, cr) in enumerate(zip(bars, compression_ratios)):
        ax1.annotate(f'{cr:.1f}×', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + test_stds[i] + 0.03),
                    ha='center', fontsize=8, color=COLORS['secondary'])
    
    ax1.axhline(y=0.1, color=COLORS['neutral'], linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(4.5, 0.12, 'Chance', fontsize=8, color=COLORS['neutral'])
    
    # Panel 2: By compression ratio (scatter)
    ax2.scatter(compression_ratios, test_accs, s=100, c=colors, edgecolor='black', linewidth=1, zorder=3)
    
    # Fit and plot trend line
    log_cr = np.log10(compression_ratios)
    z = np.polyfit(log_cr, test_accs, 2)
    p = np.poly1d(z)
    x_smooth = np.logspace(np.log10(1.5), np.log10(60), 100)
    ax2.plot(x_smooth, np.clip(p(np.log10(x_smooth)), 0, 1), 
             color=COLORS['secondary'], linestyle='--', alpha=0.7, linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Compression Ratio (Combinations / Outputs)')
    ax2.set_ylabel('Compositional Generalization')
    ax2.set_title('B. Compression Ratio Predicts Performance')
    ax2.set_ylim([0, 1.1])
    ax2.set_xlim([1, 100])
    
    # Annotate points
    for nf, cr, acc in zip(n_factors, compression_ratios, test_accs):
        ax2.annotate(f'{nf}F', xy=(cr, acc), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=COLORS['neutral'])
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, "extended_scaling.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    
    return fig


def create_asymmetry_figure(save_dir: str = "results") -> plt.Figure:
    """
    Figure 2B: Factor asymmetry effects.
    
    Shows that balanced factors are critical.
    """
    set_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Data
    configs = ['[10,10]', '[5,20]', '[4,25]', '[2,50]']
    asymmetry_ratios = [1.0, 4.0, 6.25, 25.0]
    compression_ratios = [5.0, 4.0, 3.4, 1.9]
    test_accs = [0.80, 0.304, 0.048, 0.0]
    test_stds = [0.067, 0.106, 0.047, 0.0]
    
    # Panel 1: Bar chart by configuration
    colors = [COLORS['primary'], COLORS['warning'], COLORS['danger'], COLORS['danger']]
    x_pos = np.arange(len(configs))
    bars = ax1.bar(x_pos, test_accs, yerr=test_stds, capsize=5,
                   color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    
    ax1.set_xlabel('Factor Cardinalities')
    ax1.set_ylabel('Compositional Generalization')
    ax1.set_title('A. Factor Balance is Critical')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=15)
    ax1.set_ylim([0, 1.0])
    
    # Add asymmetry ratio labels
    for bar, ar in zip(bars, asymmetry_ratios):
        height = bar.get_height()
        ax1.annotate(f'ratio={ar:.1f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height + 0.05),
                    ha='center', fontsize=8, color=COLORS['neutral'])
    
    # Panel 2: Scatter by asymmetry ratio
    ax2.scatter(asymmetry_ratios, test_accs, s=100, c=colors, 
                edgecolor='black', linewidth=1, zorder=3)
    ax2.errorbar(asymmetry_ratios, test_accs, yerr=test_stds, 
                 fmt='none', color=COLORS['neutral'], capsize=3, zorder=2)
    
    # Threshold line
    ax2.axvline(x=4, color=COLORS['danger'], linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(4.5, 0.7, 'Boundary\n(ratio ≈ 4)', fontsize=8, color=COLORS['danger'])
    
    ax2.set_xlabel('Asymmetry Ratio (max / min cardinality)')
    ax2.set_ylabel('Compositional Generalization')
    ax2.set_title('B. Performance vs Asymmetry')
    ax2.set_xlim([0, 30])
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, "extended_asymmetry.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    
    return fig


def create_noise_figure(save_dir: str = "results") -> plt.Figure:
    """
    Figure 2C: Noise robustness.
    
    Shows graceful degradation and regularization effect.
    """
    set_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Input noise data
    input_noise = [0.0, 0.1, 0.2, 0.5, 1.0]
    input_acc = [0.80, 0.68, 0.712, 0.296, 0.032]
    
    # Label noise data
    label_noise = [0.0, 0.05, 0.1, 0.2, 0.3]
    label_acc = [0.80, 0.84, 0.752, 0.608, 0.504]
    
    # Panel 1: Input noise
    ax1.plot(input_noise, input_acc, 'o-', color=COLORS['primary'], 
             linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=1)
    ax1.fill_between(input_noise, 0, input_acc, alpha=0.2, color=COLORS['primary'])
    
    ax1.axhline(y=0.1, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax1.axvline(x=0.2, color=COLORS['secondary'], linestyle=':', alpha=0.7)
    ax1.text(0.22, 0.6, 'Robust\nthreshold', fontsize=8, color=COLORS['secondary'])
    
    ax1.set_xlabel('Input Noise (Gaussian std)')
    ax1.set_ylabel('Compositional Generalization')
    ax1.set_title('A. Input Noise Robustness')
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.0])
    
    # Panel 2: Label noise
    ax2.plot(label_noise, label_acc, 's-', color=COLORS['highlight'], 
             linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=1)
    ax2.fill_between(label_noise, 0, label_acc, alpha=0.2, color=COLORS['highlight'])
    
    # Highlight regularization effect
    ax2.annotate('Regularization\neffect (+4%)', xy=(0.05, 0.84), xytext=(0.12, 0.92),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary']),
                fontsize=8, color=COLORS['primary'])
    
    ax2.axhline(y=0.1, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color=COLORS['warning'], linestyle=':', alpha=0.7)
    ax2.text(0.25, 0.52, 'Still >chance\nat 30% noise', fontsize=8, color=COLORS['warning'])
    
    ax2.set_xlabel('Label Noise (corruption probability)')
    ax2.set_ylabel('Compositional Generalization')
    ax2.set_title('B. Label Noise Robustness')
    ax2.set_xlim([0, 0.32])
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, "extended_noise.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    
    return fig


def create_architecture_figure(save_dir: str = "results") -> plt.Figure:
    """
    Figure 2D: Architecture comparison.
    
    Shows that principle holds across architectures but some help more.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data
    architectures = ['RNN', 'MLP', 'Transformer']
    bottleneck_20 = [0.408, 0.80, 0.92]
    stds = [0.148, 0.067, 0.044]
    
    colors = [COLORS['danger'], COLORS['neutral'], COLORS['primary']]
    x_pos = np.arange(len(architectures))
    
    bars = ax.bar(x_pos, bottleneck_20, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar, acc, std in zip(bars, bottleneck_20, stds):
        ax.annotate(f'{acc:.0%}', 
                   xy=(bar.get_x() + bar.get_width()/2, acc + std + 0.03),
                   ha='center', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.1, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax.text(2.3, 0.12, 'Chance', fontsize=8, color=COLORS['neutral'])
    
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Compositional Generalization')
    ax.set_title('Architecture Comparison (Bottleneck = 20)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(architectures)
    ax.set_ylim([0, 1.1])
    
    # Add insight annotation
    ax.annotate('Attention aids\ncompositionality', xy=(2, 0.92), xytext=(1.5, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary']),
               fontsize=9, color=COLORS['primary'])
    ax.annotate('Sequential processing\nhinders compositionality', xy=(0, 0.41), xytext=(0.3, 0.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['danger']),
               fontsize=9, color=COLORS['danger'])
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, "extended_architecture.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    
    return fig


def create_summary_figure(save_dir: str = "results") -> plt.Figure:
    """
    Create a comprehensive 2×2 summary figure.
    """
    set_publication_style()
    
    fig = plt.figure(figsize=(12, 10))
    
    # === Panel A: Main 2×2 result ===
    ax1 = fig.add_subplot(221)
    
    matrix = np.array([[0.0, 0.004], [0.0, 0.724]])
    im = ax1.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if matrix[i,j] < 0.4 else 'black'
            ax1.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Non-Comp\nOutput', 'Comp\nOutput'])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Holistic\nInput', 'Factorized\nInput'])
    ax1.set_title('A. Conjunctive Necessity\n(Interaction = 0.72)', fontweight='bold')
    
    # === Panel B: Scaling ===
    ax2 = fig.add_subplot(222)
    
    n_factors = [2, 3, 4, 5]
    test_accs = [0.20, 0.912, 1.0, 1.0]
    colors = [COLORS['danger'], COLORS['warning'], COLORS['primary'], COLORS['primary']]
    
    ax2.bar(n_factors, test_accs, color=colors, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Number of Factors')
    ax2.set_ylabel('Generalization')
    ax2.set_title('B. Scaling: More Factors = Better', fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.set_xticks(n_factors)
    
    # === Panel C: Asymmetry ===
    ax3 = fig.add_subplot(223)
    
    asymmetry = [1.0, 4.0, 6.25, 25.0]
    asym_acc = [0.80, 0.304, 0.048, 0.0]
    
    ax3.scatter(asymmetry, asym_acc, s=100, c=[COLORS['primary'], COLORS['warning'], 
                                               COLORS['danger'], COLORS['danger']],
               edgecolor='black', linewidth=1.5)
    ax3.axvline(x=4, color=COLORS['danger'], linestyle='--', alpha=0.7)
    ax3.set_xlabel('Asymmetry Ratio')
    ax3.set_ylabel('Generalization')
    ax3.set_title('C. Factor Balance Required', fontweight='bold')
    ax3.set_xlim([0, 30])
    ax3.set_ylim([0, 1.0])
    ax3.text(5, 0.7, 'Boundary', fontsize=9, color=COLORS['danger'])
    
    # === Panel D: Architecture ===
    ax4 = fig.add_subplot(224)
    
    archs = ['RNN', 'MLP', 'Trans']
    arch_acc = [0.408, 0.80, 0.92]
    arch_colors = [COLORS['danger'], COLORS['neutral'], COLORS['primary']]
    
    ax4.bar(archs, arch_acc, color=arch_colors, edgecolor='black', linewidth=1)
    ax4.set_xlabel('Architecture')
    ax4.set_ylabel('Generalization')
    ax4.set_title('D. Architecture Effects', fontweight='bold')
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, "extended_summary.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    
    # Also save PDF
    pdf_path = os.path.join(save_dir, "extended_summary.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    return fig


def create_all_extended_figures(save_dir: str = "results"):
    """Generate all extended experiment figures."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating extended experiment figures...")
    print("-" * 50)
    
    create_scaling_figure(save_dir)
    create_asymmetry_figure(save_dir)
    create_noise_figure(save_dir)
    create_architecture_figure(save_dir)
    create_summary_figure(save_dir)
    
    print("-" * 50)
    print("All extended figures generated!")


if __name__ == "__main__":
    create_all_extended_figures()