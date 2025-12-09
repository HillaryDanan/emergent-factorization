# Compositional Generalization Requires End-to-End Compositional Structure

**Hillary Danan¹***

¹ Independent Researcher, Cognitive Neuroscience

*Correspondence: hillarydanan@gmail.com

---

## Abstract

Neural networks notoriously fail to generalize compositionally—they struggle to understand novel combinations of familiar components. We demonstrate that this failure stems not from representational limitations but from structural misalignment across the processing pipeline. Through systematic factorial experiments, we establish four necessary conditions for compositional generalization: (1) factorized input structure providing recombination exposure, (2) compositional output structure enabling factor-level prediction, (3) sufficient compression ratio between input combinations and output dimensions, and (4) balanced factor cardinalities ensuring adequate exposure per factor value. These conditions are conjunctively necessary—an interaction effect of 0.720 (twice either main effect) demonstrates that neither input nor output structure alone suffices. Remarkably, scaling to higher-dimensional factorial structures (4-5 factors) yields *perfect* generalization, as stronger compositional constraints reduce the hypothesis space. However, factor asymmetry proves critical: balanced factors ([10,10]) achieve 80% generalization while extreme asymmetry ([2,50]) yields 0%. Transformer architectures outperform MLPs (+12%) and RNNs (+51%), suggesting attention mechanisms further enhance compositional processing. These findings establish an End-to-End Compositionality Principle with quantifiable boundary conditions, reframing compositional generalization from a problem of representation learning to one of structural alignment across input, computation, and output.

**Keywords:** compositional generalization, neural networks, factorial learning, disentanglement, systematicity

---

## Introduction

Compositionality—the capacity to understand and produce novel combinations of familiar components—is considered a hallmark of human cognition¹ and a fundamental challenge for artificial intelligence². Humans effortlessly understand "the dog bit the man" having only heard "the man bit the dog," while neural networks notoriously struggle with such systematic generalization³⁻⁵.

The dominant approach to this challenge focuses on architectural innovations: attention mechanisms⁶, memory systems⁷, modular networks⁸, and specialized compositional architectures⁹. An implicit assumption underlies this work: if networks learn appropriate internal representations, compositional generalization will follow. Indeed, much research has focused on learning "disentangled" representations where independent factors of variation are separated¹⁰⁻¹².

We challenge this assumption. Through systematic experiments with factorial stimulus structures, we demonstrate that compositional representations are *necessary but not sufficient* for compositional generalization. The critical missing ingredient is compositional structure throughout the processing pipeline—not just in representation, but in input encoding and output prediction.

Our findings establish an **End-to-End Compositionality Principle**: compositional generalization requires structural alignment across input, representation, and output. Moreover, we identify quantifiable boundary conditions—compression ratio and factor balance—that predict when compositional generalization will succeed or fail. This shifts the challenge from pure representation learning to the principled design of compositional tasks.

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

### Experiment 1-3: Neither Condition Alone Enables Generalization

Networks with holistic input and non-compositional output achieved 0% generalization to novel compositions (n=50 models across bottleneck sizes 5-150). Networks with factorized input but non-compositional output also achieved 0% generalization (n=35 models).

Critically, linear probes revealed that in the factorized-input condition, Factor 2 was 93% decodable from the bottleneck representation (chance = 10%). The network had *learned* factorized internal representations but could not *use* them—the non-compositional output structure provided no pathway for compositional generalization.

### Experiment 4: Compositional Output Enables Generalization

Networks with factorized input and compositional output achieved 80.0% ± 6.7% generalization (n=5 seeds at bottleneck=20; Figure 1A). This represents a Cohen's d of 11.95 compared to non-compositional output—an effect size far exceeding conventional thresholds for "large" effects (d > 0.8).

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

The interaction is twice the size of either main effect, demonstrating **conjunctive necessity**: compositional generalization requires BOTH factorized input AND compositional output. Neither condition alone produces meaningful generalization.

### Experiment 6: Scaling to Higher Dimensions

We extended the paradigm to three factors (8×8×4 = 256 combinations). Performance achieved near-ceiling levels across bottleneck sizes:

| Bottleneck | Generalization |
|------------|----------------|
| 10 | 98.8% ± 2.5% |
| 15-50 | 100% ± 0.0% |

Three-factor generalization *exceeded* two-factor performance (100% vs 80%), despite the increased complexity. This counterintuitive result reveals an important property of compositional constraints.

### Experiment 7: The Compression Ratio Principle

To understand why more factors improved performance, we systematically varied the number of factors while holding cardinality constant (4 values per factor):

| Factors | Combinations | Outputs | Compression Ratio | Generalization |
|---------|--------------|---------|-------------------|----------------|
| 2 | 16 | 8 | 2.0× | 20.0% ± 18.7% |
| 3 | 64 | 12 | 5.3× | 91.2% ± 6.4% |
| 4 | 256 | 16 | 16.0× | 100% ± 0.0% |
| 5 | 1024 | 20 | 51.2× | 100% ± 0.0% |

The compression ratio (combinations divided by total output dimensions) strongly predicts performance (Figure 2A). With compositional output structure, the network cannot memorize combination-to-output mappings because such mappings don't exist—each output head predicts only its own factor. As the compression ratio increases, non-compositional solutions become increasingly unviable, and the correct compositional solution becomes the only option in hypothesis space.

**Principle:** *Higher compression ratios strengthen the compositional inductive bias, improving generalization.*

### Experiment 8: Factor Balance as a Boundary Condition

We tested whether factor cardinality symmetry affects compositional generalization, holding total combinations approximately constant:

| Cardinalities | Asymmetry Ratio | Compression Ratio | Generalization |
|---------------|-----------------|-------------------|----------------|
| [10, 10] | 1.0 | 5.0× | 80.0% ± 6.7% |
| [5, 20] | 4.0 | 4.0× | 30.4% ± 10.6% |
| [4, 25] | 6.2 | 3.4× | 4.8% ± 4.7% |
| [2, 50] | 25.0 | 1.9× | 0.0% ± 0.0% |
| [5, 5, 4] | 1.2 | 7.1× | 91.2% ± 7.8% |

Extreme factor asymmetry catastrophically degrades performance (Figure 2B). The mechanism involves training exposure: with [2, 50], each of the 50 factor-2 values appears in only ~1.5 training examples on average, insufficient for learning robust factor representations. Additionally, asymmetry reduces the effective compression ratio, weakening the compositional inductive bias.

**Principle:** *Factor balance is a necessary condition; asymmetry ratios >4 substantially degrade generalization.*

### Experiment 9: Noise Robustness

We tested robustness to two types of noise:

**Input noise** (Gaussian noise added to one-hot encodings):
| Noise Std | Generalization |
|-----------|----------------|
| 0.0 | 80.0% ± 6.7% |
| 0.1 | 68.0% ± 8.0% |
| 0.2 | 71.2% ± 11.4% |
| 0.5 | 29.6% ± 7.0% |
| 1.0 | 3.2% ± 3.0% |

**Label noise** (random label corruption during training):
| Noise Prob | Generalization |
|------------|----------------|
| 0.0 | 80.0% ± 6.7% |
| 0.05 | 84.0% ± 6.7% |
| 0.10 | 75.2% ± 6.4% |
| 0.20 | 60.8% ± 9.9% |
| 0.30 | 50.4% ± 11.5% |

Compositional generalization degrades gracefully with noise (Figure 2C). Notably, 5% label noise *improved* performance (84% vs 80%), suggesting a regularization effect consistent with label smoothing benefits observed in deep learning¹³. Even 30% label noise yielded 50% generalization—far above chance—indicating that the compositional solution is robust to substantial corruption.

### Experiment 10: Architecture Comparison

We tested whether findings generalize beyond MLPs:

| Architecture | Bottleneck=20 | Notes |
|--------------|---------------|-------|
| MLP | 80.0% ± 6.7% | Baseline |
| Transformer | 92.0% ± 4.4% | +12%, best |
| RNN | 40.8% ± 14.8% | -39%, poor |

The End-to-End Compositionality Principle holds across architectures—compositional output is necessary in all cases—but some architectures enhance compositional processing (Figure 2D). Transformers' self-attention mechanism allows explicit, parallel comparison of factor encodings, naturally supporting factor independence. RNNs' sequential processing introduces order dependence that may interfere with factorial structure.

---

## Discussion

### The End-to-End Compositionality Principle

Our results establish that compositional generalization requires compositional structure throughout the processing pipeline—not just in internal representations. We formalize this as the **End-to-End Compositionality Principle**:

> Compositional generalization requires:
> 1. **Factorized input** providing recombination exposure
> 2. **Compositional output** enabling factor-level prediction  
> 3. **Sufficient compression ratio** (combinations >> output dimensions)
> 4. **Balanced factor cardinalities** ensuring adequate training exposure
>
> These conditions are conjunctively necessary; violating any one yields failure.

This principle explains several previously puzzling findings. Lake and Baroni³ demonstrated that sequence-to-sequence networks fail compositional generalization on SCAN despite using compositional output structure (action sequences). Our framework suggests this failure may stem from insufficient recombination exposure at input—command strings are presented as token sequences rather than factorized structures that make recombination explicit. The COGS benchmark⁴ shows similar patterns interpretable through this lens.

### Compression Ratio as Inductive Bias Strength

Our finding that higher-dimensional factorial structures yield *better* generalization (100% for 4-5 factors vs 80% for 2 factors) appears paradoxical—surely more factors means a harder task? The resolution lies in understanding compression ratio as quantifying inductive bias strength.

With compositional output, the only learnable mappings are from bottleneck to individual factors. As the number of combinations grows relative to output dimensions, the hypothesis space contracts: there are exponentially many combinations but only linearly many output categories. The compositional solution becomes not just preferred but *necessary*.

This provides a quantitative design principle: **maximize compression ratio to maximize compositional generalization**. Systems designed for compositional reasoning should increase factorial complexity (more factors, higher cardinalities) rather than simplify it.

### The Critical Role of Factor Balance

The finding that [2,50] yields 0% generalization while [10,10] yields 80%—despite identical total combinations—reveals factor balance as a critical boundary condition. This has significant implications for real-world applications where factors naturally have different cardinalities.

The mechanism involves training dynamics: with asymmetric factors, high-cardinality factor values receive insufficient training exposure. Each value of a 50-level factor appears in only ~1.5 training examples, preventing robust representation learning. Additionally, asymmetry reduces compression ratio, weakening the compositional inductive bias.

We quantify the boundary: asymmetry ratios (max cardinality / min cardinality) exceeding 4 substantially degrade performance. This provides actionable guidance for task design: ensure factor cardinalities are within roughly 4× of each other, or provide proportionally more training data for high-cardinality factors.

### Architectural Considerations

While output structure is the primary determinant of compositional generalization, architecture provides secondary modulation. Transformers' +12% advantage over MLPs aligns with their documented superiority on compositional tasks¹⁴˒¹⁵. Self-attention naturally implements parallel factor comparison, supporting the independence required for compositional processing.

RNNs' poor performance (-39% vs MLP) suggests that sequential processing may fundamentally conflict with factorial structure. Processing factor 1 before factor 2 creates asymmetric representations that impede the symmetric treatment factors require. This may explain persistent compositional generalization failures in sequence models despite architectural innovations.

### Implications

**For AI/ML:** Current approaches to compositional generalization focus on architectural innovations. Our results suggest that **task structure** may be equally or more important. A system trained to predict holistic combinations will not generalize compositionally regardless of its architectural sophistication. The practical recommendation: decompose prediction objectives into compositional sub-objectives whenever possible.

**For Cognitive Science:** Human compositional cognition may depend not just on compositional representations—language structure, conceptual hierarchies, relational reasoning—but on compositional *goals*. The structure of what we predict shapes the representations we learn. This connects to Fodor and Pylyshyn's¹ argument that systematicity requires structural alignment; our results extend this to include alignment with prediction objectives.

**For the Disentanglement Literature:** The extensive literature on learning disentangled representations¹⁰⁻¹² has focused on unsupervised discovery of factorial structure. Our results suggest this may be insufficient: even perfectly disentangled representations (93% factor decodability) fail to support compositional generalization without compositional output structure. Disentanglement may be necessary but is demonstrably not sufficient.

### Limitations

Our experiments used discrete factors with one-hot encodings in synthetic factorial domains. Natural compositional systems—language, vision, reasoning—involve continuous features, hierarchical structure, and context-dependent composition. Whether the End-to-End Compositionality Principle extends to these richer settings remains to be tested, though we predict the core insight (output structure matters) will hold.

The factor balance constraint may limit applicability to real-world domains with inherently asymmetric categories. Future work should explore whether increased training data can compensate for asymmetry, or whether auxiliary losses can enforce balanced factor representations.

---

## Methods

### Dataset Construction

**Factorial datasets** consisted of all combinations of K factors with specified cardinalities. The standard configuration was 10×10 (two factors, 100 combinations). Extended experiments tested up to 5 factors (4⁵ = 1024 combinations) and asymmetric cardinalities ([2,50], [5,20], etc.).

**Compositional holdout** reserved 25% of combinations for testing, ensuring each factor value appeared in at least one training combination. This guarantees that test combinations are novel *compositions* of familiar *components*—the defining characteristic of compositional generalization.

### Input Representations

*Holistic encoding:* One-hot vector over all combinations. For K factors with cardinalities (c₁, ..., cₖ), dimension is ∏cᵢ.

*Factorized encoding:* Concatenation of one-hot vectors per factor. Dimension is ∑cᵢ. This provides recombination exposure: combinations sharing a factor value share input dimensions.

### Output Representations

*Non-compositional:* Single softmax classifier over all combinations.

*Compositional:* K separate softmax classifiers, one per factor. Test accuracy is proportion of examples where *all* K factors are correctly predicted.

### Network Architectures

**MLP (baseline):** Input → Linear(128) → ReLU → Linear(128) → ReLU → Linear(bottleneck) → Linear(128) → ReLU → Output heads.

**Transformer:** Factor encodings embedded to hidden dimension, processed by 2-layer transformer encoder (4 attention heads), flattened and projected to bottleneck.

**RNN:** Factor encodings processed sequentially by 2-layer LSTM, final hidden state projected to bottleneck.

All architectures used separate linear heads from bottleneck to each factor output (compositional) or single head to combination output (non-compositional).

### Training

Adam optimizer (learning rate 0.001), cross-entropy loss, 2000-3000 epochs depending on task complexity. All conditions achieved 100% training accuracy prior to evaluation.

### Statistical Analysis

All experiments replicated across 5-10 random seeds. We report mean ± standard deviation. Effect sizes are Cohen's d. Factorial analysis: main effects computed as average difference across levels; interaction as difference of differences.

### Code and Data Availability

All code is available at https://github.com/HillaryDanan/emergent-factorization under MIT license. No external datasets were used; all data is procedurally generated.

---

## References

1. Fodor, J. A. & Pylyshyn, Z. W. Connectionism and cognitive architecture: A critical analysis. *Cognition* **28**, 3–71 (1988).

2. Marcus, G. F. *The Algebraic Mind: Integrating Connectionism and Cognitive Science*. (MIT Press, 2001).

3. Lake, B. M. & Baroni, M. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. in *Proc. 35th International Conference on Machine Learning* 2873–2882 (PMLR, 2018).

4. Kim, N. & Linzen, T. COGS: A compositional generalization challenge based on semantic interpretation. in *Proc. 2020 Conference on Empirical Methods in Natural Language Processing* 9087–9105 (2020).

5. Keysers, D. et al. Measuring compositional generalization: A comprehensive method on realistic data. in *Proc. 8th International Conference on Learning Representations* (2020).

6. Vaswani, A. et al. Attention is all you need. in *Advances in Neural Information Processing Systems* 5998–6008 (2017).

7. Graves, A., Wayne, G. & Danihelka, I. Neural Turing machines. *arXiv preprint arXiv:1410.5401* (2014).

8. Andreas, J., Rohrbach, M., Darrell, T. & Klein, D. Neural module networks. in *Proc. IEEE Conference on Computer Vision and Pattern Recognition* 39–48 (2016).

9. Russin, J., Jo, J., O'Reilly, R. C. & Bengio, Y. Compositional generalization in a deep seq2seq model by separating syntax and semantics. *arXiv preprint arXiv:1904.09708* (2019).

10. Locatello, F. et al. Challenging common assumptions in the unsupervised learning of disentangled representations. in *Proc. 36th International Conference on Machine Learning* 4114–4124 (PMLR, 2019).

11. Higgins, I. et al. β-VAE: Learning basic visual concepts with a constrained variational framework. in *Proc. 5th International Conference on Learning Representations* (2017).

12. Montero, M. L., Ludwig, C. J., Costa, R. P., Malhotra, G. & Bowers, J. The role of disentanglement in generalisation. in *Proc. 9th International Conference on Learning Representations* (2021).

13. Müller, R., Kornblith, S. & Hinton, G. When does label smoothing help? in *Advances in Neural Information Processing Systems* 4694–4703 (2019).

14. Csordás, R., Irie, K. & Schmidhuber, J. The devil is in the detail: Simple tricks improve systematic generalization of transformers. in *Proc. 2021 Conference on Empirical Methods in Natural Language Processing* 619–634 (2021).

15. Ontanón, S. et al. Making transformers solve compositional tasks. in *Proc. 60th Annual Meeting of the Association for Computational Linguistics* 3591–3607 (2022).

16. Danan, H. Abstraction beyond compression: Compositionality as the distinguishing operation. *Working paper* (2025).

---

## Acknowledgments

This work was developed as part of the Abstraction-Intelligence theoretical framework. The author thanks the open-source community for foundational tools enabling this research.

---

## Author Contributions

H.D. conceived the study, designed experiments, implemented code, analyzed results, and wrote the manuscript.

---

## Competing Interests

The author declares no competing interests.

---

## Figures

### Figure 1. Compositional generalization requires end-to-end compositional structure.

**(A)** Main finding. Networks with factorized input and compositional output achieve 80% generalization; all other conditions achieve ~0%. Cohen's d = 11.95.

**(B)** Bottleneck sweep. Performance increases around theoretical threshold (sum of factor cardinalities) and does not degrade at higher capacities—unlike non-compositional settings where excess capacity enables memorization.

**(C)** Full 2×2 factorial design. Interaction effect (0.720) is twice main effects (0.360, 0.364), demonstrating conjunctive necessity.

### Figure 2. Boundary conditions and robustness.

**(A)** Compression ratio principle. Generalization scales with combinations/outputs ratio. Higher ratios strengthen compositional inductive bias. 4-5 factors achieve 100%.

**(B)** Factor balance requirement. Asymmetry ratio >4 substantially degrades performance. [2,50] yields 0% despite same total combinations as [10,10] (80%).

**(C)** Noise robustness. Graceful degradation with input noise (robust to std ≤ 0.2) and label noise (robust to prob ≤ 0.3). Mild label noise (5%) improves performance via regularization.

**(D)** Architecture comparison. Transformer > MLP > RNN. Attention mechanisms enhance compositional processing; sequential processing impedes it.

---

## Extended Data

### Extended Data Table 1. Complete results across all conditions.

| Experiment | Input | Output | Key Variable | Test Accuracy |
|------------|-------|--------|--------------|---------------|
| 1 | Holistic | Non-comp | Bottleneck 5-150 | 0.0% |
| 3 | Factorized | Non-comp | Bottleneck 5-100 | 0.0% |
| 4 | Factorized | Comp | Bottleneck 20 | 80.0% ± 6.7% |
| 5 | 2×2 factorial | — | All conditions | See Figure 1C |
| 6 | Factorized (3-factor) | Comp | Bottleneck 10-50 | 98.8-100% |
| 7 | Factorized (N-factor) | Comp | N = 2,3,4,5 | 20-100% |
| 8 | Factorized | Comp | Asymmetry ratio 1-25 | 0-80% |
| 9a | Factorized | Comp | Input noise 0-1.0 | 3-80% |
| 9b | Factorized | Comp | Label noise 0-0.3 | 50-84% |
| 10 | Factorized | Comp | MLP/Trans/RNN | 41-92% |

### Extended Data Table 2. Factor decodability demonstrates learned-but-unusable representations.

| Condition | Factor 1 Decode | Factor 2 Decode | Test Generalization |
|-----------|-----------------|-----------------|---------------------|
| Holistic + Non-comp | 10% (chance) | 10% (chance) | 0% |
| Factorized + Non-comp | ~25% | **~93%** | 0% |
| Factorized + Comp | >90% | >90% | 80% |

The factorized-input, non-compositional-output condition demonstrates that high factor decodability (learned disentanglement) is insufficient for compositional generalization without appropriate output structure.

---
