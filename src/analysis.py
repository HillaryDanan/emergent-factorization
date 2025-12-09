"""
Analysis functions for measuring factorization and compositionality.

These metrics operationalize the theoretical constructs from APH.
See Danan (2025) "Abstraction Beyond Compression" Section 8.

Key metrics:
1. Factor decodability: Can we linearly decode each factor?
2. Representation geometry: Do same-factor items cluster?
3. Compositional generalization: Novel combination accuracy

Scientific note: These metrics are OPERATIONALIZATIONS of theoretical
constructs. They are not the constructs themselves. Validation requires
showing they predict downstream outcomes (transfer, generalization).
"""

import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from scipy import stats


def analyze_representations(
    model,
    dataset,
    device: str = "cpu"
) -> Dict:
    """
    Comprehensive analysis of bottleneck representations.
    
    Measures:
    1. Linear decodability of each factor
    2. Within-factor vs between-factor distances
    3. PCA alignment with factors
    
    Interpretation guide:
    - High factor decodability + high separation ratio = FACTORIZED
    - Low factor decodability + low separation ratio = HOLISTIC/ENTANGLED
    """
    model.eval()
    
    # Get all training representations
    inputs, _, f1_targets, f2_targets = dataset.get_batch("train")
    inputs = inputs.to(device)
    
    with torch.no_grad():
        _, representations = model(inputs)
        representations = representations.cpu().numpy()
    
    f1_targets = f1_targets.numpy()
    f2_targets = f2_targets.numpy()
    
    results = {}
    
    # ===== 1. LINEAR DECODABILITY =====
    # If factorization emerged, each factor should be linearly separable
    # Reference: Locatello et al. (2019) use similar metrics for disentanglement
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Factor 1
        clf1 = LogisticRegression(max_iter=1000, random_state=42)
        f1_scores = cross_val_score(clf1, representations, f1_targets, cv=5)
        results["factor1_decodability"] = float(np.mean(f1_scores))
        results["factor1_decodability_std"] = float(np.std(f1_scores))
        
        # Factor 2
        clf2 = LogisticRegression(max_iter=1000, random_state=42)
        f2_scores = cross_val_score(clf2, representations, f2_targets, cv=5)
        results["factor2_decodability"] = float(np.mean(f2_scores))
        results["factor2_decodability_std"] = float(np.std(f2_scores))
    
    # ===== 2. REPRESENTATION GEOMETRY =====
    # Factorized representations should cluster by factor value
    
    # Factor 1 clustering
    within_f1, between_f1 = [], []
    for i in range(len(representations)):
        for j in range(i + 1, len(representations)):
            dist = np.linalg.norm(representations[i] - representations[j])
            if f1_targets[i] == f1_targets[j]:
                within_f1.append(dist)
            else:
                between_f1.append(dist)
    
    results["within_factor1_dist"] = float(np.mean(within_f1)) if within_f1 else 0
    results["between_factor1_dist"] = float(np.mean(between_f1)) if between_f1 else 0
    results["factor1_separation_ratio"] = (
        results["between_factor1_dist"] / results["within_factor1_dist"]
        if results["within_factor1_dist"] > 0 else float('inf')
    )
    
    # Factor 2 clustering
    within_f2, between_f2 = [], []
    for i in range(len(representations)):
        for j in range(i + 1, len(representations)):
            dist = np.linalg.norm(representations[i] - representations[j])
            if f2_targets[i] == f2_targets[j]:
                within_f2.append(dist)
            else:
                between_f2.append(dist)
    
    results["within_factor2_dist"] = float(np.mean(within_f2)) if within_f2 else 0
    results["between_factor2_dist"] = float(np.mean(between_f2)) if between_f2 else 0
    results["factor2_separation_ratio"] = (
        results["between_factor2_dist"] / results["within_factor2_dist"]
        if results["within_factor2_dist"] > 0 else float('inf')
    )
    
    # ===== 3. PCA ALIGNMENT =====
    # Do principal components align with factors?
    if representations.shape[1] >= 2:
        n_components = min(representations.shape[1], 10)
        pca = PCA(n_components=n_components)
        pca_repr = pca.fit_transform(representations)
        results["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
        
        # Check correlation of top PCs with factors
        for pc_idx in range(min(3, pca_repr.shape[1])):
            pc = pca_repr[:, pc_idx]
            f1_corr = abs(np.corrcoef(pc, f1_targets)[0, 1])
            f2_corr = abs(np.corrcoef(pc, f2_targets)[0, 1])
            results[f"pc{pc_idx+1}_factor1_corr"] = float(f1_corr) if not np.isnan(f1_corr) else 0
            results[f"pc{pc_idx+1}_factor2_corr"] = float(f2_corr) if not np.isnan(f2_corr) else 0
    
    return results


def compute_compositionality_score(
    test_accuracy: float,
    factor1_decodability: float,
    factor2_decodability: float,
    factor1_separation: float,
    factor2_separation: float
) -> Dict[str, float]:
    """
    Compute aggregate compositionality metrics.
    
    Based on APH Section 8 (Measuring Compositionality).
    
    Components:
    - CGR: Compositional Generalization Rate (test accuracy on novel compositions)
    - Factor decodability: Average factor decodability
    - Separation: Average separation ratio
    
    Note: This is a working operationalization. Validation required.
    """
    # Normalize separation ratios (typically between 1 and 3)
    norm_sep1 = min(factor1_separation / 2.0, 1.0)
    norm_sep2 = min(factor2_separation / 2.0, 1.0)
    
    return {
        "cgr": test_accuracy,  # Compositional Generalization Rate
        "avg_factor_decodability": (factor1_decodability + factor2_decodability) / 2,
        "avg_separation_ratio": (norm_sep1 + norm_sep2) / 2,
        "composite_score": (
            test_accuracy * 0.5 +
            (factor1_decodability + factor2_decodability) / 2 * 0.3 +
            (norm_sep1 + norm_sep2) / 2 * 0.2
        )
    }


def statistical_comparison(
    group1: List[float],
    group2: List[float],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2"
) -> Dict:
    """
    Perform statistical comparison between two groups.
    
    Uses t-test for normally distributed data, Mann-Whitney U otherwise.
    Reports effect size (Cohen's d).
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return {
        f"{group1_name}_mean": float(mean1),
        f"{group1_name}_std": float(std1),
        f"{group2_name}_mean": float(mean2),
        f"{group2_name}_std": float(std2),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_p05": p_value < 0.05,
        "significant_p01": p_value < 0.01
    }