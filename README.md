# Emergent Factorization

**Testing whether neural networks discover factorized representations under architectural and informational pressure.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Finding

**Compositional generalization requires compositional structure END-TO-END.**

| Input Structure | Output Structure | Compositional Generalization |
|-----------------|------------------|------------------------------|
| Holistic | Non-compositional | **0%** |
| Factorized | Non-compositional | **0%** |
| Factorized | Compositional | **80% ± 6.7%** |

**Effect size: Cohen's d = 11.95**

![Main Results](results/main_results.png)

## The Discovery

We set out to test whether capacity pressure alone could induce factorized representations in neural networks. The answer was *no*—but the investigation revealed something more important:

> **Output structure acts as an inductive bias that constrains the solution space.**
>
> Non-compositional output (100-way classification over combinations) permits memorization.
> Compositional output (separate factor predictions) FORCES compositional solutions.

This refines the Abstraction Primitive Hypothesis (APH): the conditions for compositional abstraction apply to the **entire processing pipeline**, not just representation learning.

## Theoretical Background

This repository tests predictions from the **Abstraction Primitive Hypothesis** (APH):

> **Hypothesis**: Compression yields abstraction (factorized, compositional representations) when specific conditions are met: factorization pressure, recombination exposure, compositional bottlenecks, and multi-task learning.

*Source: Danan (2025), "Abstraction Beyond Compression", Section 9.2*

### Key References

- Compositional generalization failures in neural networks: Lake & Baroni (2018), ICML
- Disentanglement requires inductive biases: Locatello et al. (2019), ICML  
- Information bottleneck: Tishby et al. (2000)
- Systematicity and compositionality: Fodor & Pylyshyn (1988)

## Experiments

### Experiment 1-2: Capacity and Multi-Task Pressure (Null Results)

**Question**: Does bottleneck size or multi-task learning induce factorization?

**Result**: 0% compositional generalization regardless of bottleneck size or training objective.

**Diagnosis**: The input representation (one-hot over combinations) provided no recombination signal.

### Experiment 3: Input Structure (Null Result)

**Question**: Does factorized input representation enable generalization?

**Result**: 0% compositional generalization despite factorized inputs.

**Diagnosis**: The output structure (100-way combination classification) was non-compositional.

**Key Observation**: Factor 2 was ~93% decodable from the bottleneck! The network *learned* factorized representations but couldn't use them because the output structure didn't permit composition.

### Experiment 4: Compositional Output (KEY RESULT)

**Question**: Does compositional output structure enable generalization?

**Result**: **80% ± 6.7% compositional generalization** (Cohen's d = 11.95)

**Design**: 
- Input: Factorized (concatenated factor one-hots)
- Output: Compositional (separate factor prediction heads)
- Test: Both factors correct on novel combinations

### Bottleneck Sweep Results

| Bottleneck | Test Accuracy | Notes |
|------------|---------------|-------|
| 5 | 31.2% ± 11.7% | Underfitting |
| 10 | 51.2% ± 8.2% | Constrained |
| 15 | 72.8% ± 3.9% | Near threshold |
| **20** | **80.0% ± 6.7%** | **At threshold (10+10)** |
| 25 | 74.4% ± 13.5% | High variance |
| 30 | 67.2% ± 15.7% | High variance |
| 50 | 84.8% ± 12.0% | Excess capacity |

**Notable**: Unlike non-compositional output, we do NOT see degradation at high bottleneck sizes. Compositional output structure prevents memorization strategies.

## Theoretical Implications

### Refinement of APH Section 9.2

**Original conditions** for compositional abstraction:
1. Factorization pressure
2. Recombination exposure  
3. Compositional bottleneck
4. Multi-task learning

**Refined understanding**: These conditions apply to the **entire processing pipeline**:

> **End-to-End Compositionality Principle**: Compositional generalization requires compositional structure at input (recombination exposure), representation (factorization pressure), AND output (compositional prediction structure).

### Why Output Structure Matters

With non-compositional output (combination classification):
- The decoder maps bottleneck → arbitrary combination index
- Novel combinations have no decoder pathway
- Even perfect factor representations cannot generalize

With compositional output (factor prediction):
- The decoder maps bottleneck → factor 1 AND bottleneck → factor 2
- Novel combinations use the SAME factor decoders
- Compositional structure is preserved end-to-end

## Usage
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run main experiment (Experiment 4)
python3 -m src.run_all

# Generate figures
python3 -m src.visualize

# Run specific experiments
python3 -m src.experiment_compositional_output
python3 -m src.experiment_input_structure
python3 -m src.experiment_bottleneck
```

## Project Structure
```
emergent-factorization/
├── src/
│   ├── data.py                      # Holistic input dataset
│   ├── data_factorized.py           # Factorized input dataset
│   ├── models.py                    # Network architectures
│   ├── experiment_bottleneck.py     # Exp 1: Capacity pressure
│   ├── experiment_multitask.py      # Exp 2: Multi-task pressure
│   ├── experiment_input_structure.py # Exp 3: Input structure
│   ├── experiment_compositional_output.py # Exp 4: Output structure
│   ├── analysis.py                  # Representation analysis
│   ├── visualize.py                 # Figure generation
│   ├── utils.py                     # Utilities
│   └── run_all.py                   # Main entry point
├── results/                         # Experiment outputs
├── requirements.txt
└── README.md
```

## Remaining Questions

1. **Holistic input + compositional output**: Does output structure alone enable generalization? (Experiment 5, in progress)

2. **3+ factors**: Does this scale to more complex factorial structures?

3. **Learned compositional structure**: Can networks learn to impose compositional output?

4. **Connection to language**: How does this relate to compositional semantics?

## Author

**Hillary Danan, PhD**  
Cognitive Neuroscience

## Related Work

This is part of the [Abstraction-Intelligence](https://github.com/HillaryDanan/Abstraction-Intelligence) theoretical framework.

## License

MIT

---

*"Compositional generalization requires compositional structure—not just in representation, but end-to-end."*