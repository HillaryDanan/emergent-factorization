# Emergent Factorization

**Compositional generalization requires compositional structure end-to-end: Evidence from factorial learning in neural networks.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Finding: The End-to-End Compositionality Principle

Compositional generalization requires **four conjunctively necessary conditions**:

1. **Factorized input** (recombination exposure)
2. **Compositional output** (factor-level prediction)
3. **Sufficient compression ratio** (combinations >> outputs)
4. **Balanced factor cardinalities** (asymmetry ratio < ~2.5)

Violating any one yields degraded or failed generalization. Simple compensation strategies (oversampling, loss reweighting) do not overcome violated conditions.

---

## Main Results

### Conjunctive Necessity (Experiments 1-5)

|                    | Non-Compositional Output | Compositional Output |
|--------------------|--------------------------|----------------------|
| **Holistic Input**     | 0.0% ± 0.0%              | 0.4% ± 1.2%          |
| **Factorized Input**   | 0.0% ± 0.0%              | **72.4% ± 12.7%**    |

- **Interaction effect: 0.720** (twice either main effect)
- **Cohen's d = 11.95** (compositional vs non-compositional output)

### Scaling (Experiments 6-7)

| Factors | Combinations | Compression Ratio | Generalization |
|---------|--------------|-------------------|----------------|
| 2 | 16 | 2.0× | 20% |
| 3 | 64 | 5.3× | 91% |
| 4 | 256 | 16× | **100%** |
| 5 | 1024 | 51× | **100%** |

**Finding:** Higher compression ratios strengthen the compositional inductive bias.

### Factor Balance Boundary (Experiment 8)

| Asymmetry Ratio | Performance | Status |
|-----------------|-------------|--------|
| 1.0 | 80% | ✓ Reliable |
| 1.5 | 65% | ✓ Reliable |
| 2.0 | 62% | ✓ Reliable |
| **2.5-2.8** | **~50%** | **Boundary** |
| 4.0 | 30% | ✗ Degraded |
| 6.25 | 5% | ✗ Severe |
| 11.0 | 0% | ✗ Failed |

**Finding:** Boundary for reliable generalization lies at asymmetry ratio ~2.5.

### Asymmetry vs Compression Trade-off (Experiment 9)

Both effects are substantial and nearly equal:
- **Asymmetry effect:** 49.6 percentage points
- **Compression effect:** 46.2 percentage points

**Key finding:** Asymmetry can be partially compensated by compression. [10,40] achieves 77% despite ratio 4.0.

### Compensation Strategies Fail (Experiment 10)

| Strategy | Performance | Change |
|----------|-------------|--------|
| Baseline | 30.4% | — |
| Oversample 2× | 21.6% | -8.8% |
| Oversample 4× | 20.0% | -10.4% |
| Balanced loss | **0.8%** | **-29.6%** |

**Finding:** All compensation strategies make things worse. The constraint is fundamental, not a training artifact.

### Noise Robustness (Experiment 11)

- **Input noise:** Robust to std ≤ 0.2
- **Label noise:** Robust to prob ≤ 0.3; 5% noise may help (regularization)

### Architecture Comparison (Experiment 12)

| Architecture | Performance |
|--------------|-------------|
| Transformer | 92.0% ± 4.4% |
| MLP | 80.0% ± 6.7% |
| RNN | 40.8% ± 14.8% |

- Transformer vs MLP: **t(6.9) = 3.35, p = 0.013, Cohen's d = 2.12**
- Attention mechanisms enhance compositional processing
- Sequential processing (RNN) hinders it

---

## Quantified Boundary Conditions

| Condition | Threshold | Evidence |
|-----------|-----------|----------|
| Compression ratio | >5× for reliable | Scaling experiment |
| Asymmetry ratio | <2.5 for reliable | Boundary sweep |
| Interaction | Both can compensate | Matched controls |

---

## Theoretical Implications

### For AI/ML
Task structure may be as important as architecture. Decompose prediction objectives into compositional sub-objectives.

### For Disentanglement
Disentangled representations (93% factor decodability) are **necessary but not sufficient**—compositional output structure is required.

### For Cognitive Science
Human compositional cognition may depend not just on compositional representations but on compositional *goals*.

---

## Repository Structure

```
emergent-factorization/
├── src/
│   ├── data.py                           # Holistic input dataset
│   ├── data_factorized.py                # Factorized input dataset
│   ├── models.py                         # Network architectures
│   ├── experiment_bottleneck.py          # Exp 1: Capacity pressure
│   ├── experiment_multitask.py           # Exp 2: Multi-task pressure
│   ├── experiment_input_structure.py     # Exp 3: Input structure
│   ├── experiment_compositional_output.py # Exp 4: Output structure (KEY)
│   ├── experiment_holistic_compositional.py # Exp 5: Full factorial
│   ├── experiment_three_factors.py       # Exp 6: 3-factor scaling
│   ├── experiment_extended.py            # Exp 7-12: Extended suite
│   ├── experiment_asymmetry_boundary.py  # Asymmetry deep-dive
│   ├── analysis.py                       # Representation analysis
│   ├── visualize.py                      # Core figures
│   ├── visualize_extended.py             # Extended figures
│   ├── utils.py                          # Utilities
│   └── run_all.py                        # Main entry point
├── results/                              # Experiment outputs
├── paper_draft_v3.md                     # Paper manuscript
├── requirements.txt
└── README.md
```

---

## Usage

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run core experiments
python3 -m src.run_all

# Run extended experiments
python3 -m src.experiment_extended

# Run asymmetry boundary investigation
python3 -m src.experiment_asymmetry_boundary

# Generate figures
python3 -m src.visualize
python3 -m src.visualize_extended
```

---

## Key References

- Lake & Baroni (2018). Generalization without systematicity. *ICML*.
- Locatello et al. (2019). Challenging common assumptions in disentanglement. *ICML*.
- Fodor & Pylyshyn (1988). Connectionism and cognitive architecture. *Cognition*.
- Keysers et al. (2020). Measuring compositional generalization. *ICLR*.

---

## Citation

```bibtex
@misc{danan2025emergent,
  author = {Danan, Hillary},
  title = {Emergent Factorization: Compositional Generalization 
           Requires End-to-End Compositional Structure},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/emergent-factorization}
}
```

---

## Author

**Hillary Danan, PhD**  
Cognitive Neuroscience

Part of the [Abstraction-Intelligence](https://github.com/HillaryDanan/abstraction-intelligence) theoretical framework.

---

## License

MIT

---

*"Compositional generalization requires compositional structure—not just in representation, but end-to-end."*