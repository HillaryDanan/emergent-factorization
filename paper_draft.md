# Compositional Generalization Requires End-to-End Compositional Structure

**Hillary Danan¹***

¹ Independent Researcher, Cognitive Neuroscience

*Correspondence: hillarydanan@gmail.com

---

## Abstract

Neural networks notoriously fail to generalize compositionally—they struggle to understand novel combinations of familiar components. We demonstrate that this failure stems not from representational limitations but from a mismatch between learned representations and output structure. Through systematic factorial experiments, we show that compositional generalization requires compositional structure at both input AND output—neither alone suffices. Networks with factorized inputs but non-compositional outputs (classification over combinations) achieve 0% generalization to novel compositions despite learning factorized internal representations. The same networks with compositional outputs (separate factor predictions) achieve 72-100% generalization. This interaction effect (0.720) is twice the size of either main effect, demonstrating conjunctive necessity. Remarkably, scaling to three factors yields *perfect* (100%) generalization, as stronger compositional constraints reduce the hypothesis space. These findings establish an End-to-End Compositionality Principle: compositional generalization requires structural alignment across the entire processing pipeline—input, representation, and output. This reframes the compositional generalization challenge from a problem of representation learning to one of task structure design.

---

## Introduction

Compositionality—the ability to understand and produce novel combinations of familiar components—is considered a hallmark of human cognition¹ and a key challenge for artificial intelligence². Despite impressive progress in neural network capabilities, systematic compositional generalization remains elusive³⁻⁵.

The dominant approach to this challenge focuses on architectural innovations: attention mechanisms⁶, memory systems⁷, modular networks⁸, and specialized compositional architectures⁹. An implicit assumption underlies this work: if networks learn the right internal representations, compositional generalization will follow.

We challenge this assumption. Through systematic experiments with factorial stimulus structures, we demonstrate that compositional representations are *necessary but not sufficient* for compositional generalization. The critical missing ingredient is compositional structure in the output space—the task the network is trained to perform.

Our findings establish an **End-to-End Compositionality Principle**: compositional generalization requires structural alignment across input, representation, and output. This shifts the challenge from pure representation learning to the joint design of representations and tasks.

---

## Results

### Experimental Paradigm

We constructed factorial stimulus sets where inputs are combinations of independent factors (e.g., 10 colors × 10 shapes = 100 combinations). Training included most combinations; testing evaluated generalization to held-out combinations where each factor appeared in training but the specific combination did not (compositional holdout).

We manipulated two independent variables in a 2×2 factorial design:

**Input structure:**
- *Holistic*: One-hot encoding over all combinations (100 dimensions)
- *Factorized*: Concatenated one-hot encodings per factor (10 + 10 = 20 dimensions)

**Output structure:**
- *Non-compositional*: Classification over all combinations (100-way)
- *Compositional*: Separate classification per factor (10-way + 10-way)

All networks were multi-layer perceptrons with a bottleneck layer, trained to convergence (100% training accuracy in all conditions).

### Neither Condition Alone Enables Generalization

We first tested each condition in isolation. Networks with holistic input and non-compositional output achieved 0% generalization to novel compositions (Experiment 1; n=50 models across bottleneck sizes 5-150). Networks with factorized input and non-compositional output also achieved 0% generalization (Experiment 3; n=35 models).

Critically, in Experiment 3, linear probes revealed that Factor 2 was 93% decodable from the bottleneck representation (chance = 10%). The network had *learned* factorized representations but could not *use* them—the non-compositional output structure provided no pathway for compositional generalization.

### Compositional Output Enables Generalization

Networks with factorized input and compositional output achieved 80.0% ± 6.7% generalization (Experiment 4; n=5 seeds at bottleneck=20). This represents a Cohen's d of 11.95 compared to non-compositional output—an extraordinary effect size.

Performance varied with bottleneck capacity (Figure 1B):

| Bottleneck | Generalization |
|------------|----------------|
| 5 | 31.2% ± 11.7% |
| 10 | 51.2% ± 8.2% |
| 15 | 72.8% ± 3.9% |
| 20 (threshold) | 80.0% ± 6.7% |
| 50 | 84.8% ± 12.0% |

The theoretical threshold for factorized representation is the sum of factor cardinalities (10 + 10 = 20). Performance increased sharply around this threshold but did not degrade at higher capacities—unlike non-compositional settings where excess capacity enables memorization.

### Full Factorial Design Reveals Conjunctive Necessity

To isolate the contributions of input and output structure, we conducted a complete 2×2 factorial experiment (Experiment 5; n=10 seeds per condition):

|                    | Non-Compositional Output | Compositional Output |
|--------------------|--------------------------|----------------------|
| **Holistic Input**     | 0.0% ± 0.0%              | 0.4% ± 1.2%          |
| **Factorized Input**   | 0.0% ± 0.0%              | 72.4% ± 12.7%        |

Statistical analysis revealed:
- Main effect of input structure: 0.360
- Main effect of output structure: 0.364
- **Interaction effect: 0.720**

The interaction is twice the size of either main effect. This demonstrates **conjunctive necessity**: compositional generalization requires BOTH factorized input AND compositional output. Neither condition alone produces any meaningful generalization.

### Scaling to Higher Dimensions

We extended the paradigm to three factors (8 colors × 8 shapes × 4 sizes = 256 combinations; Experiment 6; n=5 seeds per bottleneck size):

| Bottleneck | Generalization (all 3 factors correct) |
|------------|----------------------------------------|
| 10 | 98.8% ± 2.5% |
| 15 | 100% ± 0.0% |
| 20 (threshold) | 100% ± 0.0% |
| 30 | 100% ± 0.0% |
| 50 | 100% ± 0.0% |

Three-factor generalization exceeded two-factor performance, achieving near-perfect accuracy even below the theoretical threshold. This counterintuitive result—more factors yielding better generalization—reveals an important property of compositional constraints.

The compression ratio (combinations / output categories) provides insight:
- 2-factor: 100 combinations / 20 outputs = 5×
- 3-factor: 256 combinations / 20 outputs = 12.8×

Higher compression ratios make the compositional solution increasingly *necessary*, reducing the viable hypothesis space and paradoxically simplifying learning.

---

## Discussion

### The End-to-End Compositionality Principle

Our results establish that compositional generalization requires compositional structure throughout the processing pipeline—not just in internal representations but in input encoding and output prediction. We term this the **End-to-End Compositionality Principle**.

This principle explains several previously puzzling findings. Lake and Baroni³ demonstrated that sequence-to-sequence networks fail compositional generalization on the SCAN benchmark despite using compositional output structure (action sequences). Our framework suggests this failure may stem from insufficient compositional structure at input—the command strings, while compositional, are presented as sequences of tokens rather than factorized structures that make recombination explicit.

The principle also explains why certain architectural innovations help: attention mechanisms⁶ create compositional *processing* (parallel factor extraction); modular networks⁸ enforce compositional *computation* (separate modules for separate factors). Both approaches impose compositional structure beyond just input and output.

### Why Output Structure Matters

With non-compositional output (combination classification), the decoder learns arbitrary mappings from bottleneck representations to combination indices. Novel combinations have no established pathway—even if the bottleneck perfectly encodes the constituent factors.

With compositional output (factor prediction), each decoder head learns to extract *one* factor from the bottleneck. Novel combinations use the *same* factor decoders, enabling systematic generalization.

This framing recasts compositional generalization as a problem of **structural alignment** between representation and task, not representation learning alone.

### The Paradox of Stronger Constraints

The finding that three-factor generalization exceeds two-factor generalization appears paradoxical—surely more factors means a harder task? The resolution lies in understanding compositional constraints as *inductive biases* that reduce hypothesis space.

With compositional output structure, the network cannot memorize combination-to-output mappings because such mappings don't exist—each output head predicts only its own factor. The only viable solution is factor extraction. As the number of combinations grows relative to the number of output categories, alternative (non-compositional) solutions become increasingly unviable, and the correct solution becomes easier to find.

This suggests a design principle: **stronger compositional constraints aid generalization**. Systems designed for compositional generalization should maximize the ratio of compositional complexity to output dimensionality.

### Implications for AI

Current approaches to compositional generalization focus heavily on architectural innovations. Our results suggest that **task structure** may be equally important. A system trained to predict holistic combinations will not generalize compositionally regardless of its architectural sophistication.

This has practical implications for training regimes, curriculum design, and evaluation protocols. Compositional generalization may require not just the right architecture but the right *objective*—one that decomposes into compositional sub-objectives.

### Implications for Cognitive Science

Human compositional cognition may depend not just on compositional representations—language, conceptual structure, hierarchical planning—but on compositional *goals*. The structure of what we predict shapes the representations we learn.

This connects to Fodor and Pylyshyn's¹ argument that systematicity requires structural alignment between representation and operation. Our results provide empirical support and extend the argument: alignment must include the structure of prediction itself.

### Limitations

Our experiments used simple factorial structures with discrete factors and one-hot encodings. Natural compositional domains (language, vision, reasoning) involve continuous features, hierarchical structure, and context-dependent composition. Whether the End-to-End Compositionality Principle extends to these settings remains to be tested.

Additionally, we tested feed-forward networks with bottleneck architectures. Recurrent networks, transformers, and other architectures may show different patterns—though we predict the basic principle (output structure matters) will hold.

---

## Methods

### Dataset Construction

**Two-factor datasets** consisted of all combinations of 10 values on Factor 1 (e.g., colors) and 10 values on Factor 2 (e.g., shapes), yielding 100 combinations. Compositional holdout reserved 25% of combinations for testing, ensuring each factor value appeared in at least one training combination.

**Three-factor datasets** consisted of 8 × 8 × 4 = 256 combinations with analogous holdout procedure.

### Input Representations

*Holistic*: One-hot encoding over all combinations. For two factors, this is a 100-dimensional vector with a single 1.

*Factorized*: Concatenation of one-hot encodings per factor. For two factors with cardinality 10 each, this is a 20-dimensional vector (two 10-dimensional one-hots concatenated).

### Output Representations

*Non-compositional*: Single classification head over all combinations (softmax over 100 classes for two factors).

*Compositional*: Separate classification heads per factor. For two factors, this is two 10-way classifiers. Test accuracy is the proportion of examples where *all* factors are correctly predicted.

### Network Architecture

All networks were multi-layer perceptrons: input → 128-unit hidden layer (ReLU) → 128-unit hidden layer (ReLU) → bottleneck layer (variable size) → 128-unit hidden layer (ReLU) → output.

Networks with compositional output had separate decoder heads branching from the bottleneck.

### Training

All networks were trained with Adam optimizer (learning rate 0.001) for 2000 epochs (two-factor) or 3000 epochs (three-factor) using cross-entropy loss. Training continued until 100% training accuracy was achieved in all conditions.

### Statistical Analysis

All experiments were replicated across 5-10 random seeds. We report mean ± standard deviation. Effect sizes are Cohen's d. Factorial analysis computed main effects as average difference across levels of each factor; interaction as the difference of differences.

### Code Availability

All code is available at https://github.com/HillaryDanan/emergent-factorization under MIT license.

---

## References

1. Fodor, J. A. & Pylyshyn, Z. W. Connectionism and cognitive architecture: A critical analysis. *Cognition* **28**, 3–71 (1988).

2. Marcus, G. F. *The Algebraic Mind: Integrating Connectionism and Cognitive Science*. (MIT Press, 2001).

3. Lake, B. M. & Baroni, M. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. in *Proceedings of the 35th International Conference on Machine Learning* 2873–2882 (PMLR, 2018).

4. Kim, N. & Linzen, T. COGS: A compositional generalization challenge based on semantic interpretation. in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* 9087–9105 (2020).

5. Keysers, D. et al. Measuring compositional generalization: A comprehensive method on realistic data. in *Proceedings of the 8th International Conference on Learning Representations* (2020).

6. Vaswani, A. et al. Attention is all you need. in *Advances in Neural Information Processing Systems* 5998–6008 (2017).

7. Graves, A., Wayne, G. & Danihelka, I. Neural Turing machines. *arXiv preprint arXiv:1410.5401* (2014).

8. Andreas, J., Rohrbach, M., Darrell, T. & Klein, D. Neural module networks. in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* 39–48 (2016).

9. Russin, J., Jo, J., O'Reilly, R. C. & Bengio, Y. Compositional generalization in a deep seq2seq model by separating syntax and semantics. *arXiv preprint arXiv:1904.09708* (2019).

10. Locatello, F. et al. Challenging common assumptions in the unsupervised learning of disentangled representations. in *Proceedings of the 36th International Conference on Machine Learning* 4114–4124 (PMLR, 2019).

11. Montero, M. L., Ludwig, C. J., Costa, R. P., Malhotra, G. & Bowers, J. The role of disentanglement in generalisation. in *Proceedings of the 9th International Conference on Learning Representations* (2021).

12. Higgins, I. et al. β-VAE: Learning basic visual concepts with a constrained variational framework. in *Proceedings of the 5th International Conference on Learning Representations* (2017).

13. Tishby, N., Pereira, F. C. & Bialek, W. The information bottleneck method. *arXiv preprint physics/0004057* (2000).

14. Danan, H. Abstraction beyond compression: Compositionality as the distinguishing operation. *Working paper* (2025).

---

## Acknowledgments

This work was developed as part of the Abstraction-Intelligence theoretical framework. The author thanks the scientific community for foundational work on compositionality and representation learning.

---

## Author Contributions

H.D. conceived the study, designed experiments, implemented code, analyzed results, and wrote the manuscript.

---

## Competing Interests

The author declares no competing interests.

---

## Figures

### Figure 1. Compositional generalization requires end-to-end compositional structure.

**(A)** Effect of output structure. Networks with factorized input but non-compositional output (red) achieve 0% generalization to novel compositions. Networks with factorized input and compositional output (green) achieve 80.0% ± 6.7% generalization. Cohen's d = 11.95.

**(B)** Bottleneck sweep with compositional output. Performance increases sharply around the theoretical threshold (sum of factor cardinalities = 20) and does not degrade at higher capacities.

**(C)** Full 2×2 factorial design. Generalization requires BOTH factorized input AND compositional output. Interaction effect (0.720) is twice the main effects (0.360, 0.364).

### Figure 2. Compositional generalization scales to higher dimensions.

**(A)** Three-factor experiment (8 × 8 × 4 = 256 combinations) achieves near-perfect generalization (100%) across bottleneck sizes, exceeding two-factor performance.

**(B)** Higher compression ratios (combinations / outputs) make compositional solutions increasingly necessary, paradoxically improving generalization.

---

## Extended Data

### Extended Data Table 1. Complete results by condition.

| Experiment | Input | Output | Bottleneck | Train Acc | Test Acc | n |
|------------|-------|--------|------------|-----------|----------|---|
| 1 | Holistic | Non-comp | 5-150 | 100% | 0.0% | 50 |
| 3 | Factorized | Non-comp | 5-100 | 100% | 0.0% | 35 |
| 4 | Factorized | Comp | 20 | 100% | 80.0% ± 6.7% | 5 |
| 5A | Holistic | Non-comp | 20 | 100% | 0.0% ± 0.0% | 10 |
| 5B | Holistic | Comp | 20 | 100% | 0.4% ± 1.2% | 10 |
| 5C | Factorized | Non-comp | 20 | 100% | 0.0% ± 0.0% | 10 |
| 5D | Factorized | Comp | 20 | 100% | 72.4% ± 12.7% | 10 |
| 6 | Factorized (3-factor) | Comp | 10-50 | 100% | 98.8-100% | 35 |

### Extended Data Table 2. Factor decodability from bottleneck representations.

| Condition | Factor 1 Decodability | Factor 2 Decodability |
|-----------|----------------------|----------------------|
| Holistic + Non-comp | 10% (chance) | 10% (chance) |
| Factorized + Non-comp | ~25% | **~93%** |
| Factorized + Comp | >90% | >90% |

Note: Factorized input with non-compositional output shows high Factor 2 decodability despite 0% compositional generalization, demonstrating that learned representations are necessary but not sufficient.

---
