# False Confidence in Generative Samplers: Exact Physical References Expose Hidden Errors and Diagnostic Blindness

[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15011779%2Fv1-blue.svg)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011779/v1)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**Official repository** for the manuscript:  
*False Confidence in Generative Samplers: Exact Physical References Expose Hidden Errors and Diagnostic Blindness*  
**Author:** Karthik Gangadharaiah (`Independent researcher (KoWerk), Amsterdam, The Netherlands`)  
**Preprint:** [ChemRxiv: 10.26434/chemrxiv.15011779/v1](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011779/v1)  
**Journal Status:** Under Review at *Nature Machine Intelligence* (`NATMACHINTELL-A26116821`)  
**Manuscript PDF:** [`paper/main.pdf`](paper/main.pdf) (12 pages, TMLR format, de-anonymized preprint)

---

## Overview

Learned samplers (normalizing flows, discrete diffusion, GFlowNets) are increasingly proposed for equilibrium Boltzmann sampling and free-energy estimation in chemistry and materials science. In practice, these models are evaluated almost exclusively by self-normalized diagnostics—principally the **Effective Sample Size (ESS)** and nominal delta-method confidence intervals.

This repository provides the mathematical proof and empirical code demonstrating that **self-normalized diagnostics suffer from a complete blind spot for unvisited support**:

```
                       [ Unvisited Support Pathology ]

     Target Boltzmann Law p(x)                  Learned Proposal q(x)
     +---------------------------+              +---------------------------+
     | Basin A     |  Basin B    |              | Basin A     |  Basin B    |
     | (Visited)   | (Unvisited) |              | (Visited)   | (DROPPED)   |
     | 1 - alpha   |    alpha    |              | 100% Mass   |   0% Mass   |
     +---------------------------+              +---------------------------+
                   |                                          |
                   v                                          v
      True Free Energy: F                      Estimated Free Energy: F_hat
                                               Bias: -k_B T ln(1 - alpha)
                                               Nominal ESS: > 90% (False Confidence)
                                               95% CI Coverage: 0% Asymptotic
```

### Core Contributions
1. **The Diagnostic Blindness Theorem (Proposition 1)**: Conditioned on failing to propose points in a basin of Boltzmann mass $\alpha$, importance weights are distributed identically to the truncated problem. The free-energy estimate asymptotically targets $F - \beta^{-1}\ln(1-\alpha)$, while normalized weights and $\text{ESS}$ are mathematically invariant to $\alpha$.
2. **Exact Disordered Benchmark ($n = 24$)**: On solvable Sherrington-Kirkpatrick spin glasses with 1.68 × 10⁷ configurations evaluated without sampling error, discrete diffusion and GFlowNets collapse to **8% and 47% coverage** at $\beta = 2$, while reporting high internal confidence.
3. **Continuous Molecular Flow Audit**: A RealNVP Boltzmann Generator trained via reverse-KL on 2D Alanine Dipeptide collapses into $C_{7\rm eq}$, proposing 0.000% of points in $C_{7\rm ax}$ ($\alpha = 8.1\%$). Despite reporting **79% internal ESS**, nominal 95% interval coverage collapses to **10%**.
4. **All-Atom Molecular Dynamics**: 300 ns Amber14SB trajectories of Chignolin and Trp-cage demonstrate that dropped native states incur $0.68$ to $1.02\,$kcal/mol thermodynamic shifts—well exceeding the $0.3\,$kcal/mol lead-optimization threshold.
5. **The Bidirectional Diagnostic**: A practical cross-audit ($G_{\rm rev}$) evaluating reference configurations against the proposal. On the Alanine Dipeptide flow, while the forward Jensen gap is blind ($0.028\,$nats), $G_{\rm rev}$ registers **11.66 nats** (with $C_{7\rm ax}$ accounting for 88% of the gap), immediately exposing the omission.

---

## Repository Structure

```
├── paper/                      # Complete LaTeX source & compiled preprint
│   ├── main.tex                # Root manuscript file (12 pages, TMLR format)
│   ├── main.pdf                # Compiled publication preprint (matches ChemRxiv)
│   ├── sections/               # Modular section files (intro, theory, benchmarks...)
│   ├── tables/                 # Benchmark audit tables (tab_chemical_audit.tex)
│   ├── figures/                # Vector PDFs and PNGs for Figures 1 to 6
│   └── generated/              # LaTeX values.tex & values.json (100% data-derived)
├── data/                       # Stored simulation & benchmark artifacts
│   ├── summary_sk24.json       # Exact n=24 spin-glass test-split ledger (70 records)
│   ├── alanine_flow_audit.json # 100-run RealNVP flow evaluation metrics
│   ├── chignolin_overnight_metrics.npz # 30,000-frame RMSD and Rg metrics (300 ns)
│   ├── trpcage_overnight_metrics.npz   # 30,000-frame RMSD and Rg metrics (300 ns)
│   └── lj38_inversion_results.json     # Lennard-Jones LJ38 Langevin sweep
├── scripts/                    # Reproduction & audit scripts
│   ├── plotting.py             # Typewriter Lab visual style theme
│   ├── make_figures.py         # Master script generating all figures, tables & macros
│   └── eval_alanine_flow.py    # Standalone RealNVP Boltzmann generator audit
├── requirements.txt            # Minimal Python dependencies
├── pyproject.toml              # Project specification
└── LICENSE                     # Apache 2.0 License
```

---

## Quickstart & Reproduction

### 1. Installation
Clone the repository and install the minimal dependencies:
```bash
git clone https://github.com/QntmSeer/false-confidence-generative-samplers.git
cd false-confidence-generative-samplers
pip install -r requirements.txt
```

### 2. Regenerate All Manuscript Figures & Numbers
To regenerate all 6 figures, the audit table, and `paper/generated/values.tex` directly from stored data:
```bash
python scripts/make_figures.py
```
Outputs are saved directly to `paper/figures/` and `paper/generated/values.tex`.

### 3. Run the Alanine Dipeptide RealNVP Flow Audit
To train the RealNVP Boltzmann generator via reverse-KL on the 2D Ramachandran landscape and audit its failure:
```bash
python scripts/eval_alanine_flow.py
```
This runs 100 evaluation batches ($M=5{,}000$ proposals each), measures true error against exact 2D numerical quadrature ($300 \times 300$ grid), calculates nominal coverage, and evaluates the bidirectional check ($G_{\rm rev}$ vs $G_{\rm fwd}$).

### 4. Compile Manuscript
```bash
cd paper
pdflatex -interaction=batchmode main.tex
bibtex main
pdflatex -interaction=batchmode main.tex
pdflatex -interaction=batchmode main.tex
```

---

## Citation

If you use this benchmark, code, or diagnostic check in your research, please cite the ChemRxiv preprint:

```bibtex
@article{gangadharaiah2026falseconfidence,
  author    = {Gangadharaiah, Karthik},
  title     = {False Confidence in Generative Samplers: Exact Physical References Expose Hidden Errors and Diagnostic Blindness},
  journal   = {ChemRxiv},
  year      = {2026},
  doi       = {10.26434/chemrxiv.15011779/v1},
  url       = {https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011779/v1}
}
```

---

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
