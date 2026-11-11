# False Confidence in Generative Samplers: Exact Physical References Expose Hidden Errors and Diagnostic Blindness

[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15011779%2Fv2-blue.svg)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011779/v2)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**Official repository** for the manuscript:  
*False Confidence in Generative Samplers: Exact Physical References Expose Hidden Errors and Diagnostic Blindness*  
**Author:** Karthik Gangadharaiah (`Independent researcher (KoWerk), Amsterdam, The Netherlands`)  
**Preprint:** [ChemRxiv: 10.26434/chemrxiv.15011779/v2](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011779/v2)  
**Manuscript PDF:** [`paper/main.pdf`](paper/main.pdf) (v3, 14 pages, TMLR format, de-anonymized preprint)  
**Changes from v2:** [`CHANGES_v3.md`](CHANGES_v3.md)

---

## Overview

Learned samplers (normalizing flows, discrete diffusion, GFlowNets) are increasingly proposed for equilibrium Boltzmann sampling and free-energy estimation in chemistry and materials science. In practice, these models are evaluated almost exclusively by self-normalized diagnostics, principally the **Effective Sample Size (ESS)** and nominal delta-method confidence intervals.

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
                                               Nominal ESS: can stay near 100%
                                               95% CI Coverage: -> 0% while B is unvisited
```

### Core Contributions
1. **The Diagnostic Blindness Theorem (Proposition 1)**: Conditioned on failing to propose points in a basin of Boltzmann mass $\alpha$, importance weights are distributed identically to the truncated problem. The free-energy estimate asymptotically targets $F - \beta^{-1}\ln(1-\alpha)$, while normalized weights and $\text{ESS}$ are mathematically invariant to $\alpha$.
2. **Exact Disordered Benchmark ($n = 24$)**: On solvable Sherrington-Kirkpatrick spin glasses with 1.68 × 10⁷ configurations evaluated without sampling error, discrete diffusion and GFlowNets collapse to **8% and 47% coverage** at $\beta = 2$, while reporting high internal confidence.
3. **Continuous Flow Audit**: A RealNVP flow trained via reverse-KL on an analytic 2D surrogate of the alanine dipeptide Ramachandran surface collapses into $C_{7\rm eq}$, proposing 0.000% of points in $C_{7\rm ax}$ ($\alpha = 8.1\%$). Despite reporting **79% internal ESS**, nominal 95% interval coverage collapses to **10%**.
4. **All-Atom Molecular Dynamics**: Amber14SB/OBC2 trajectories of chignolin (1 µs) and Trp-cage (300 ns) size the effect: omitting the Trp-cage native basin (50% of the run) would shift the free energy by about 0.42 kcal/mol, above a 0.3 kcal/mol target. The trajectories are not equilibrated, and the literature populations quoted in the paper are unverified.
5. **The Bidirectional Diagnostic**: A practical cross-audit ($G_{\rm rev}$) evaluating reference configurations against the proposal. On the Alanine Dipeptide flow, while the forward Jensen gap is blind ($0.028\,$nats), $G_{\rm rev}$ registers **11.66 nats**. On model landscapes up to 414 dimensions (two analytic surrogates and the real LJ38 potential restricted to two basins) it behaves the same way, but one control with the correct support also gives a large gap, so a large $G_{\rm rev}$ is not specific to a missing basin.

---

## Repository Structure

```
├── paper/                      # Complete LaTeX source & compiled preprint
│   ├── main.tex                # Root manuscript file (14 pages, TMLR format)
│   ├── main.pdf                # Compiled preprint (v3)
│   ├── sections/               # Section files (intro, theory, benchmarks, ...)
│   ├── tables/                 # Generated tables (tab_chemical_audit.tex, tab_3d_flows.tex)
│   ├── figures/                # Vector PDFs and PNGs for Figures 1 to 7
│   └── generated/              # values.tex & values.json: every data-derived number in the text
├── data/                       # Stored simulation & benchmark artifacts
│   ├── summary_sk24.json       # Exact n=24 spin-glass test-split summary (70 records)
│   ├── alanine_flow_audit.json # 2D alanine-surrogate RealNVP flow, 100 runs
│   ├── molecular_3d_flow_audit.json # Flows on the 66D/114D/408D model landscapes (k-hat notes inside)
│   ├── chignolin_1us_metrics.npz    # 100,000-frame RMSD and Rg (1 us chignolin run)
│   ├── chignolin_overnight_metrics.npz # 30,000-frame RMSD and Rg (earlier 300 ns chignolin run)
│   ├── trpcage_overnight_metrics.npz   # 30,000-frame RMSD and Rg (300 ns Trp-cage run)
│   ├── chignolin_ca_conformations.json # C-alpha coordinates for the figure insets
│   ├── lj38_inversion_results.json     # LJ38 Langevin sweep
│   ├── *.pdb, *.xyz            # Reference structures (1UAO, alanine, LJ38 minima)
│   └── chignolin_1us_audit.json # LEGACY, not used by the paper: its ESS values are random placeholders
├── scripts/                    # Reproduction & audit scripts
│   ├── plotting.py             # Figure style
│   ├── make_figures.py         # Generates all figures, tables & text macros from data/
│   ├── eval_alanine_flow.py    # 2D alanine-surrogate RealNVP flow audit (CPU)
│   ├── eval_3d_molecular_flows.py      # Flows on the higher-dimensional model landscapes (GPU)
│   └── eval_3d_equivariant_and_khat.py # SE(3) flow and Pareto k-hat (GPU)
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
To regenerate all 7 figures, both tables, and `paper/generated/values.tex` directly from stored data:
```bash
python scripts/make_figures.py
```
Outputs are saved directly to `paper/figures/` and `paper/generated/values.tex`.

### 3. Run the Alanine Dipeptide RealNVP Flow Audit
To train the RealNVP flow via reverse KL on the analytic 2D Ramachandran surrogate and audit its failure:
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
  doi       = {10.26434/chemrxiv.15011779/v2},
  url       = {https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011779/v2}
}
```

---

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
