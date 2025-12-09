# Compositional Generalization Requires End-to-End Compositional Structure

**Hillary Danan¹***

¹ Independent Researcher, Cognitive Neuroscience

*Correspondence: hillarydanan@gmail.com

---

## Abstract

Neural networks notoriously fail to generalize compositionally—struggling to understand novel combinations of familiar components. We demonstrate that this failure stems from structural misalignment across the processing pipeline, not representational limitations. Through systematic factorial experiments, we establish four necessary conditions for compositional generalization: (1) factorized input structure providing recombination exposure, (2) compositional output structure enabling factor-level prediction, (3) sufficient compression ratio between combinations and output dimensions, and (4) balanced factor cardinalities. These conditions are conjunctively necessary—an interaction effect of 0.720 (twice either main effect) demonstrates that neither input nor output structure alone suffices. Scaling to higher-dimensional factorial structures (4-5 factors) yields perfect generalization, as stronger compositional constraints reduce the hypothesis space. However, factor asymmetry proves critical: the boundary lies at asymmetry ratio ~2.5, sharper than expected, and simple compensation strategies (oversampling, loss reweighting) fail to overcome it—suggesting this constraint is fundamental rather than a training artifact. Transformer architectures outperform MLPs by 12 percentage points (p = 0.013), while RNNs underperform by 39 points, suggesting attention mechanisms enhance compositional processing. These findings establish an End-to-End Compositionality Principle with quantifiable boundary conditions, reframing compositional generalization from representation learning to structural alignment across input, computation, and output.

**Keywords:** compositional generalization, neural networks, factorial learning, disentanglement, systematicity

---

## Introduction

Compositionality—the capacity to understand and produce novel combinations of familiar components—is a hallmark of human cognition¹ and a fundamental challenge for artificial intelligence². Humans effortlessly understand "the dog bit the man" having only heard "the man bit the dog," while neural networks notoriously struggle with such systematic generalization³⁻⁵.

The dominant approach focuses on architectural innovations: attention mechanisms⁶, memory systems⁷, modular networks⁸, and specialized compositional architectures⁹. An implicit assumption underlies this work: if networks learn appropriate internal representations, compositional generalization will follow. Indeed, much research has focused on learning "disentangled" representations where independent factors of variation are separated¹⁰⁻¹².

We challenge this assumption. Through systematic experiments with factorial stimulus structures, we demonstrate that compositional representations are *necessary but not sufficient* for compositional generalization. The critical missing ingredient is compositional structure throughout the processing pipeline—not just in representation, but in input encoding and output prediction.

Our findings establish an **End-to-End Compositionality Principle**: compositional generalization requires structural alignment across input, representation, and output. Moreover, we identify quantifiable boundary conditions—compression ratio and factor balance—that predict when compositional generalization will succeed or fail. Importantly, simple strategies to compensate for violated conditions (oversampling, loss reweighting) fail, suggesting these constraints are fundamental rather than artifacts of training dynamics.

---

## Results

### Experimental Paradigm

We constructed factorial stimulus sets where inputs are combinations of independent factors (e.g., 10 colors × 10 shapes = 100 combinations). Training included most combinations; testing evaluated generalization to held-out combinations where each factor value appeared in training but the specific combination did not (compositional holdout; see Methods).

We manipulated two independent variables in a 2×2 factorial design:

**Input structure:**
- *Holistic*: One-hot encoding over all combinations (100 dimensions for 10×10)
- *Factorized*: Concatenated one-hot encodings per factor (20 dimensions for 10×10)

**Output structure:**
- *Non-compositional*: Classification over all combinations (100-way)
- *Compositional*: Separate classification per factor (10-way + 10-way)

All networks were multi-layer perceptrons with bottleneck layers, trained to convergence (100% training accuracy in all conditions).

### Experiments 1-3: Neither Condition Alone Enables Generalization

Networks with holistic input and non-compositional output achieved 0% generalization to novel compositions (n=50 models across bottleneck sizes 5-150). Networks with factorized input but non-compositional output also achieved 0% generalization (n=35 models).

Critically, linear probes revealed that in the factorized-input condition, Factor 2 was 93% decodable from the bottleneck representation (chance = 10%). The network had *learned* factorized internal representations but could not *use* them—the non-compositional output structure provided no pathway for compositional generalization.

### Experiment 4: Compositional Output Enables Generalization

Networks with factorized input and compositional output achieved 80.0% ± 6.7% generalization (n=5 seeds at bottleneck=20; Figure 1A). This represents a Cohen's d of 11.95 compared to non-compositional output—an effect size far exceeding conventional thresholds.

### Experiment 5: Conjunctive Necessity

A complete 2×2 factorial experiment (n=10 seeds per condition) isolated the contributions of input and output structure (Figure 1C):

|                    | Non-Compositional Output | Compositional Output |
|--------------------|--------------------------|----------------------|
| **Holistic Input**     | 0.0% ± 0.0%              | 0.4% ± 1.2%          |
| **Factorized Input**   | 0.0% ± 0.0%              | 72.4% ± 12.7%        |

Statistical decomposition revealed:
- Main effect of input structure: 0.360
- Main effect of output structure: 0.364
- **Interaction effect: 0.720**

The interaction is twice the size of either main effect, demonstrating **conjunctive necessity**: compositional generalization requires BOTH factorized input AND compositional output.

### Experiment 6: Scaling to Higher Dimensions

Three-factor extension (8×8×4 = 256 combinations) achieved near-ceiling performance:

| Bottleneck | Generalization |
|------------|----------------|
| 10 | 98.8% ± 2.5% |
| 15-50 | 100% ± 0.0% |

Three-factor generalization *exceeded* two-factor performance (100% vs 80%), despite increased complexity.

### Experiment 7: The Compression Ratio Principle

To understand why more factors improved performance, we varied the number of factors while holding cardinality constant (4 values per factor):

| Factors | Combinations | Compression Ratio | Generalization |
|---------|--------------|-------------------|----------------|
| 2 | 16 | 2.0× | 20.0% ± 18.7% |
| 3 | 64 | 5.3× | 91.2% ± 6.4% |
| 4 | 256 | 16.0× | 100% ± 0.0% |
| 5 | 1024 | 51.2× | 100% ± 0.0% |

The compression ratio (combinations divided by output dimensions) strongly predicts performance (Figure 2A). With compositional output, non-compositional solutions become increasingly unviable as compression increases, forcing the network toward the correct compositional solution.

**Principle:** *Higher compression ratios strengthen the compositional inductive bias.*

### Experiment 8: Factor Balance Boundary Condition

We tested whether factor cardinality symmetry affects compositional generalization:

| Cardinalities | Asymmetry Ratio | Generalization |
|---------------|-----------------|----------------|
| [10, 10] | 1.0 | 80.0% ± 6.7% |
| [8, 12] | 1.5 | 65.0% ± 18.4% |
| [7, 14] | 2.0 | 61.7% ± 13.3% |
| [6, 17] | 2.8 | 49.6% ± 16.9% |
| [5, 20] | 4.0 | 30.4% ± 10.6% |
| [4, 25] | 6.25 | 4.8% ± 4.7% |
| [3, 33] | 11.0 | 0.0% ± 0.0% |

The boundary for reliable compositional generalization (>50%) lies at **asymmetry ratio ~2.5** (Figure 2B). Performance degrades gradually from ratio 1.0 to 2.5, then sharply from 2.5 to 6.0, reaching floor at ratio ~10.

Notably, per-factor analysis revealed asymmetric degradation: the high-cardinality factor suffers disproportionately. At [3, 33], Factor 1 (cardinality 3) maintained 51% accuracy while Factor 2 (cardinality 33) collapsed to 0%.

### Experiment 9: Isolating Asymmetry vs Compression Effects

To disentangle asymmetry and compression ratio effects, we conducted matched-control experiments:

**Group A** (matched compression ~5×, different asymmetry):
- [10, 10] (ratio 1.0): 80.0%
- [5, 20] (ratio 4.0): 30.4%
- Asymmetry effect: **49.6 percentage points**

**Group B** (matched asymmetry ~4×, different compression):
- [5, 20] (compression 4.0×): 30.4%
- [7, 28] (compression 5.6×): 49.4%
- [10, 40] (compression 8.0×): 76.6%
- Compression effect: **46.2 percentage points**

Both effects are substantial and nearly equal in magnitude. Importantly, [10, 40] demonstrates that **asymmetry can be partially compensated by increased compression ratio**—the same asymmetry ratio (4.0) that yields 30% with 100 combinations yields 77% with 400 combinations.

### Experiment 10: Compensation Strategies Fail

We tested whether simple interventions could overcome the asymmetry constraint on [5, 20]:

| Strategy | Performance | Change from Baseline |
|----------|-------------|---------------------|
| Baseline | 30.4% | — |
| Oversample rare values 2× | 21.6% | -8.8% |
| Oversample rare values 4× | 20.0% | -10.4% |
| Balanced loss weighting | 0.8% | -29.6% |
| Both (2×) | 1.6% | -28.8% |

**All compensation strategies degraded performance. Balanced loss weighting was catastrophic.**

Per-factor analysis revealed the mechanism: balanced loss (which weights the high-cardinality factor's loss more heavily) caused Factor 2 to collapse from 40% to 0.8% accuracy. The intervention intended to help the disadvantaged factor instead destroyed its learning entirely.

This negative result is scientifically important: **the factor balance constraint appears fundamental, not a training artifact amenable to simple fixes.** The asymmetry problem is not insufficient data for rare values—oversampling doesn't help. It's not unbalanced gradients—reweighting makes things worse. The constraint appears to reflect deeper properties of compositional learning.

### Experiment 11: Noise Robustness

We tested robustness to input noise (Gaussian) and label noise (random corruption):

**Input noise:**
| Noise Std | Generalization |
|-----------|----------------|
| 0.0 | 80.0% ± 6.7% |
| 0.1 | 68.0% ± 8.0% |
| 0.2 | 71.2% ± 11.4% |
| 0.5 | 29.6% ± 7.0% |

**Label noise:**
| Noise Prob | Generalization |
|------------|----------------|
| 0.0 | 80.0% ± 6.7% |
| 0.05 | 84.0% ± 6.7% |
| 0.10 | 75.2% ± 6.4% |
| 0.30 | 50.4% ± 11.5% |

Compositional generalization is robust to moderate noise. The improvement at 5% label noise (84% vs 80%) may reflect a regularization effect, though this interpretation is tentative. Even 30% label corruption yields above-chance performance, indicating the compositional solution is robust to substantial noise.

### Experiment 12: Architecture Comparison

We tested generalization beyond MLPs:

| Architecture | Performance (bottleneck=20) |
|--------------|----------------------------|
| MLP | 80.0% ± 6.7% |
| Transformer | 92.0% ± 4.4% |
| RNN | 40.8% ± 14.8% |

Welch's t-test comparing Transformer to MLP: t(6.9) = 3.35, p = 0.013, Cohen's d = 2.12, confirming the 12-point advantage is statistically significant.

The End-to-End Compositionality Principle holds across architectures—compositional output is necessary in all cases—but architecture provides secondary modulation. Transformers' self-attention allows parallel factor comparison, naturally supporting factorial structure. RNNs' sequential processing introduces order dependence that may interfere with the symmetric treatment factors require.

---

## Discussion

### The End-to-End Compositionality Principle

Our results establish that compositional generalization requires compositional structure throughout the processing pipeline. We formalize this as the **End-to-End Compositionality Principle**:

> Compositional generalization requires:
> 1. **Factorized input** providing recombination exposure
> 2. **Compositional output** enabling factor-level prediction  
> 3. **Sufficient compression ratio** (combinations >> output dimensions)
> 4. **Balanced factor cardinalities** (asymmetry ratio < ~2.5 for reliable performance)
>
> These conditions are conjunctively necessary; violating any one yields degraded or failed generalization. Simple compensation strategies do not overcome violated conditions.

### Quantified Boundary Conditions

Unlike prior work establishing that compositional generalization is difficult, we provide **quantified thresholds**:

- **Compression ratio**: >5× for reliable generalization; >15× approaches ceiling
- **Asymmetry ratio**: <2.5 for reliable generalization; >6 yields near-zero performance
- **Interaction**: Asymmetry can be partially compensated by compression ([10,40] achieves 77% despite ratio 4.0)

These thresholds enable principled task design rather than trial-and-error.

### Why Compensation Fails

The failure of oversampling and loss reweighting to overcome asymmetry constraints is theoretically significant. These interventions address surface-level explanations (insufficient data, unbalanced gradients) but fail completely—indeed, make things worse.

We hypothesize that the asymmetry constraint reflects the geometry of compositional representations. With [5, 20], the bottleneck must allocate capacity to represent both factors. Optimal allocation would be ~20% for Factor 1, ~80% for Factor 2 (proportional to information content). But the compositional output heads assume *equal* factor extractability. This architectural mismatch cannot be overcome by data augmentation or loss weighting—it requires architectural solutions (e.g., asymmetric bottleneck allocation, factor-specific capacity).

### Implications for Existing Benchmarks

Our framework generates predictions for existing compositional generalization benchmarks. For instance, our framework suggests that the failures observed on SCAN³ and COGS⁴ may reflect insufficient compositional structure at input or misaligned output structure, rather than purely architectural limitations. Testing these predictions directly remains for future work.

### Implications for AI and Cognitive Science

**For AI/ML:** Task structure may be as important as architecture. A system trained to predict holistic combinations will not generalize compositionally regardless of sophistication. The practical recommendation: decompose prediction objectives into compositional sub-objectives; ensure factor balance; maximize compression ratio.

**For cognitive science:** Human compositional cognition may depend not just on compositional representations but on compositional *goals*. The structure of what we predict shapes the representations we learn.

**For the disentanglement literature:** Even perfectly disentangled representations (93% factor decodability) fail to support compositional generalization without compositional output structure. Disentanglement is necessary but demonstrably not sufficient.

### Limitations

Our experiments used discrete factors with one-hot encodings in synthetic factorial domains. Natural compositional systems—language, vision, reasoning—involve continuous features, hierarchical structure, and context-dependent composition. Whether the End-to-End Compositionality Principle extends to these richer settings remains to be tested.

The factor balance constraint (ratio < 2.5) may limit applicability to real-world domains with inherently asymmetric categories. Our finding that increased compression can partially compensate suggests a path forward, but architectural solutions may be required for highly asymmetric domains.

---

## Methods

### Dataset Construction

**Factorial datasets** consisted of all combinations of K factors with specified cardinalities. The standard configuration was 10×10 (100 combinations). Extended experiments tested up to 5 factors and asymmetric cardinalities.

**Compositional holdout** reserved 25% of combinations for testing, ensuring each factor value appeared in at least one training combination.

### Input and Output Representations

*Holistic input:* One-hot vector over all combinations.

*Factorized input:* Concatenation of one-hot vectors per factor.

*Non-compositional output:* Single softmax classifier over all combinations.

*Compositional output:* K separate softmax classifiers, one per factor. Test accuracy is proportion where *all* K factors are correctly predicted.

### Architectures

**MLP:** Input → [Linear(128), ReLU]×2 → Linear(bottleneck) → [Linear(128), ReLU] → Output heads.

**Transformer:** Factor one-hots embedded, processed by 2-layer transformer encoder (4 heads), projected to bottleneck.

**RNN:** Factor one-hots processed sequentially by 2-layer LSTM, final hidden state projected to bottleneck.

### Compensation Strategies

**Oversampling:** Combinations containing rare factor values (appearing in <3 training examples) replicated 2× or 4×.

**Balanced loss:** Factor losses weighted by inverse cardinality (so high-cardinality factors receive higher weight), normalized to sum to K.

### Statistical Analysis

All experiments replicated across 5-10 seeds. Effect sizes are Cohen's d. Architecture comparisons use Welch's t-test. Factorial decomposition: main effects as average difference across levels; interaction as difference of differences.

### Code Availability

All code available at https://github.com/HillaryDanan/emergent-factorization (MIT license).

---

## References

1. Fodor, J. A. & Pylyshyn, Z. W. Connectionism and cognitive architecture: A critical analysis. *Cognition* **28**, 3–71 (1988).

2. Marcus, G. F. *The Algebraic Mind: Integrating Connectionism and Cognitive Science*. (MIT Press, 2001).

3. Lake, B. M. & Baroni, M. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *ICML* 2873–2882 (2018).

4. Kim, N. & Linzen, T. COGS: A compositional generalization challenge based on semantic interpretation. *EMNLP* 9087–9105 (2020).

5. Keysers, D. et al. Measuring compositional generalization: A comprehensive method on realistic data. *ICLR* (2020).

6. Vaswani, A. et al. Attention is all you need. *NeurIPS* 5998–6008 (2017).

7. Graves, A., Wayne, G. & Danihelka, I. Neural Turing machines. *arXiv:1410.5401* (2014).

8. Andreas, J., Rohrbach, M., Darrell, T. & Klein, D. Neural module networks. *CVPR* 39–48 (2016).

9. Russin, J., Jo, J., O'Reilly, R. C. & Bengio, Y. Compositional generalization in a deep seq2seq model by separating syntax and semantics. *arXiv:1904.09708* (2019).

10. Locatello, F. et al. Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML* 4114–4124 (2019).

11. Higgins, I. et al. β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR* (2017).

12. Montero, M. L., Ludwig, C. J., Costa, R. P., Malhotra, G. & Bowers, J. The role of disentanglement in generalisation. *ICLR* (2021).

13. Müller, R., Kornblith, S. & Hinton, G. When does label smoothing help? *NeurIPS* 4694–4703 (2019).

14. Csordás, R., Irie, K. & Schmidhuber, J. The devil is in the detail: Simple tricks improve systematic generalization of transformers. *EMNLP* 619–634 (2021).

15. Ontanón, S. et al. Making transformers solve compositional tasks. *ACL* 3591–3607 (2022).

---

## Acknowledgments

This work was developed as part of the Abstraction-Intelligence theoretical framework.

## Author Contributions

H.D. conceived the study, designed experiments, implemented code, analyzed results, and wrote the manuscript.

## Competing Interests

The author declares no competing interests.

---

## Figures

### Figure 1. Compositional generalization requires end-to-end compositional structure.

**(A)** Main finding. Networks with factorized input and compositional output achieve 80% generalization; all other conditions achieve ~0%. Cohen's d = 11.95.

**(B)** Bottleneck sweep. Performance increases around theoretical threshold (sum of factor cardinalities).

**(C)** Full 2×2 factorial design. Interaction effect (0.720) is twice main effects, demonstrating conjunctive necessity.

### Figure 2. Boundary conditions and robustness.

**(A)** Compression ratio principle. Generalization scales with combinations/outputs ratio. 4-5 factors achieve 100%.

**(B)** Factor balance boundary. Performance degrades gradually from ratio 1.0 to 2.5, then sharply to ratio 6.0. Boundary for reliable generalization (~50%) lies at ratio ~2.5.

**(C)** Compensation strategies fail. Oversampling and balanced loss weighting do not overcome asymmetry—balanced loss is catastrophic (0.8%).

**(D)** Architecture comparison. Transformer (92%) > MLP (80%, p=0.013) > RNN (41%).

### Figure 3. Asymmetry vs compression trade-off.

Matched controls reveal asymmetry (49.6 pp effect) and compression (46.2 pp effect) are approximately equal in magnitude. Asymmetry can be partially compensated by increased compression: [10,40] achieves 77% despite ratio 4.0.

---

## Extended Data

### Extended Data Table 1. Complete asymmetry sweep results.

| Cardinalities | Asymmetry | Compression | Test Acc | F1 Acc | F2 Acc |
|---------------|-----------|-------------|----------|--------|--------|
| [10, 10] | 1.00 | 5.00 | 80.0% | 86.4% | 87.2% |
| [8, 12] | 1.50 | 4.80 | 65.0% | 72.5% | 75.8% |
| [7, 14] | 2.00 | 4.67 | 61.7% | 75.8% | 72.5% |
| [6, 17] | 2.83 | 4.43 | 49.6% | 60.8% | 56.8% |
| [5, 20] | 4.00 | 4.00 | 30.4% | 55.2% | 40.0% |
| [5, 25] | 5.00 | 4.17 | 18.7% | 41.3% | 25.8% |
| [4, 25] | 6.25 | 3.45 | 4.8% | 39.2% | 7.2% |
| [3, 33] | 11.00 | 2.75 | 0.0% | 50.8% | 0.0% |

Note: High-cardinality factor (F2) degrades disproportionately at extreme asymmetry.

### Extended Data Table 2. Compensation strategy details.

| Strategy | Test Acc | F1 Acc | F2 Acc | Training Samples |
|----------|----------|--------|--------|------------------|
| Baseline | 30.4% | 55.2% | 40.0% | 75 |
| Oversample 2× | 21.6% | 32.8% | 44.0% | ~150 |
| Oversample 4× | 20.0% | 26.4% | 42.4% | ~300 |
| Balanced loss | 0.8% | 56.8% | 0.8% | 75 |
| Both 2× | 1.6% | 52.0% | 1.6% | ~150 |
| Both 4× | 2.4% | 48.8% | 2.4% | ~300 |

Note: Balanced loss destroys F2 learning despite weighting its loss more heavily.

---
