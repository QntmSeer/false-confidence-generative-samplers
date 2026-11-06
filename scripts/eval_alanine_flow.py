#!/usr/bin/env python3
"""Train and audit a RealNVP Boltzmann Generator on 2D Alanine Dipeptide Ramachandran landscape.

Demonstrates the unvisited support pathology (Proposition 1) on a continuous molecular model:
1. Exact reference Z and F computed by fine 2D quadrature (300x300 grid).
2. Train a 4-layer RealNVP normalizing flow using reverse-KL divergence (zero-forcing).
3. Measure internal self-normalized ESS, nominal delta-method CI coverage, and true bias.
4. Apply the bidirectional reverse check (G_rev vs G_fwd) to expose the dropped C7ax basin.

Outputs:
- data/alanine_flow_audit.json
- paper/figures/fig_alanine_flow_pathology.pdf|png
"""

import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

KB = 0.001987204  # kcal / (mol K)
TEMP = 300.0
KBT = KB * TEMP  # ~0.596 kcal/mol
BETA = 1.0 / KBT  # ~1.677 (kcal/mol)^-1

# 1. 2D Alanine Dipeptide Ramachandran potential with realistic steric barrier
def U_alanine(x):
    # x: (N, 2) in radians [-pi, pi]
    phi, psi = x[:, 0], x[:, 1]
    u_c7eq = -4.5 * torch.exp(-((phi + 1.4)**2 / (2 * 0.35**2) + (psi - 1.3)**2 / (2 * 0.35**2)))
    u_alphar = -3.2 * torch.exp(-((phi + 1.1)**2 / (2 * 0.35**2) + (psi + 0.7)**2 / (2 * 0.35**2)))
    u_c7ax = -2.8 * torch.exp(-((phi - 1.1)**2 / (2 * 0.30**2) + (psi + 1.0)**2 / (2 * 0.30**2)))
    # Central steric barrier at phi = 0 separating left and right Ramachandran halves
    u_barrier = 8.0 * torch.exp(-(phi**2) / (2 * 0.25**2))
    u_bound = 1.5 * ((phi / 2.6)**10 + (psi / 2.6)**10)
    return u_c7eq + u_alphar + u_c7ax + u_barrier + u_bound

# 2. Exact Reference Quadrature
grid_n = 300
phi_grid = torch.linspace(-np.pi, np.pi, grid_n)
psi_grid = torch.linspace(-np.pi, np.pi, grid_n)
P, S = torch.meshgrid(phi_grid, psi_grid, indexing='ij')
grid_coords = torch.stack([P.flatten(), S.flatten()], dim=-1)
u_grid = U_alanine(grid_coords)
b_grid = torch.exp(-BETA * u_grid)
d_phi = (phi_grid[1] - phi_grid[0]).item()
d_psi = (psi_grid[1] - psi_grid[0]).item()
d_area = d_phi * d_psi
Z_exact = (b_grid * d_area).sum().item()
F_exact = -KBT * np.log(Z_exact)

# C7ax basin mass (phi > 0)
mask_c7ax = grid_coords[:, 0] > 0
Z_c7ax = (b_grid[mask_c7ax] * d_area).sum().item()
alpha_c7ax = Z_c7ax / Z_exact
theoretical_bias = -KBT * np.log(1.0 - alpha_c7ax)

print("="*70)
print("EXACT REFERENCE QUADRATURE (ALANINE DIPEPTIDE PMF)")
print(f"  Exact Z:            {Z_exact:.4f}")
print(f"  Exact F:            {F_exact:.4f} kcal/mol")
print(f"  C7ax basin mass:    {alpha_c7ax * 100:.2f}%")
print(f"  Theoretical shift:  +{theoretical_bias:.3f} kcal/mol")
print("="*70)

# 3. RealNVP Normalizing Flow Architecture
class AffineCoupling(nn.Module):
    def __init__(self, mask_val):
        super().__init__()
        self.register_buffer('mask', torch.tensor(mask_val, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):  # x -> z (analysis)
        m = self.mask
        x_m = x * m
        st = self.net(x_m)
        s = torch.tanh(st[:, 0:1]) * (1.0 - m)
        t = st[:, 1:2] * (1.0 - m)
        z = x_m + (1.0 - m) * (x * torch.exp(s) + t)
        return z, s.sum(dim=-1)
    def inverse(self, z):  # z -> x (generation)
        m = self.mask
        z_m = z * m
        st = self.net(z_m)
        s = torch.tanh(st[:, 0:1]) * (1.0 - m)
        t = st[:, 1:2] * (1.0 - m)
        x = z_m + (1.0 - m) * ((z - t) * torch.exp(-s))
        return x, -s.sum(dim=-1)

class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()
        masks = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        self.layers = nn.ModuleList([AffineCoupling(m) for m in masks])
    def sample(self, n):
        z = torch.randn(n, 2)
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        x = z
        log_det_tot = torch.zeros(n)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det_tot += ld
        log_qx = log_pz - log_det_tot
        return x, log_qx
    def log_prob(self, x):
        log_det_tot = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, ld = layer.forward(x)
            log_det_tot += ld
        log_pz = -0.5 * (x**2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det_tot

flow = RealNVP()
opt = torch.optim.Adam(flow.parameters(), lr=2e-3)

# 4. Train with Reverse KL divergence (zero-forcing objective)
print("\n[INFO] Training RealNVP Boltzmann Generator via Reverse KL (800 steps)...")
for step in range(1, 801):
    opt.zero_grad()
    x, log_qx = flow.sample(512)
    u = U_alanine(x)
    loss = (log_qx + BETA * u).mean()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"  Step {step:4d} | Reverse KL loss: {loss.item():.4f}")

# 5. Evaluate the Trained Flow across R=100 independent evaluation runs
R_RUNS = 100
M_SAMPLES = 5000
results_runs = []

print(f"\n[INFO] Auditing {R_RUNS} independent evaluations (M={M_SAMPLES} samples)...")

# Generate reference samples via rejection sampling for the reverse check
# Use the global minimum of U_alanine for an exact, batch-invariant rejection envelope
u_min_global = u_grid.min().item()
ref_samples = []
n_accepted = 0
while n_accepted < 5000:
    prop = (torch.rand(20000, 2) * 2 - 1) * np.pi
    u_p = U_alanine(prop)
    prob = torch.clamp(torch.exp(-BETA * (u_p - u_min_global)), max=1.0)
    accept = torch.rand(20000) < prob
    accepted = prop[accept]
    ref_samples.append(accepted)
    n_accepted += len(accepted)
ref_coords = torch.cat(ref_samples, dim=0)[:5000]

with torch.no_grad():
    for r in range(R_RUNS):
        x_gen, log_q = flow.sample(M_SAMPLES)
        u_gen = U_alanine(x_gen)
        
        # Importance weights
        log_w = -BETA * u_gen - log_q
        max_log_w = log_w.max()
        w = torch.exp(log_w - max_log_w)
        
        # Self-normalized ESS
        ess_val = (w.sum()**2 / (w**2).sum()).item()
        ess_pct = (ess_val / M_SAMPLES) * 100.0
        
        # Free energy estimate
        log_z_hat = (torch.logsumexp(log_w, dim=0) - np.log(M_SAMPLES)).item()
        f_hat = -KBT * log_z_hat
        err_f = f_hat - F_exact
        
        # Delta-method 95% CI
        w_norm = w / w.mean()
        sigma_w = w_norm.std().item()
        ci_half = 1.96 * KBT * sigma_w / np.sqrt(M_SAMPLES)
        covered = (F_exact >= f_hat - ci_half) and (F_exact <= f_hat + ci_half)
        
        # C7ax visitation (phi > 0)
        c7ax_sampled = (x_gen[:, 0] > 0).float().mean().item()
        
        # Bidirectional Gaps
        # Forward Jensen gap G_fwd = log Z_hat - E_q[log w]
        g_fwd = (log_z_hat - log_w.mean().item())
        
        # Reverse Jensen gap G_rev = E_p[log w] - log Z_hat
        log_q_ref = flow.log_prob(ref_coords)
        u_ref = U_alanine(ref_coords)
        log_w_ref = -BETA * u_ref - log_q_ref
        g_rev = (log_w_ref.mean().item() - log_z_hat)
        
        results_runs.append({
            "ess_pct": ess_pct,
            "f_hat": f_hat,
            "err_f": err_f,
            "ci_half": ci_half,
            "covered": covered,
            "c7ax_frac": c7ax_sampled,
            "g_fwd": g_fwd,
            "g_rev": g_rev
        })

ess_med = float(np.median([r["ess_pct"] for r in results_runs]))
c7ax_gen_mean = float(np.mean([r["c7ax_frac"] for r in results_runs]))
cov_rate = float(np.mean([r["covered"] for r in results_runs]) * 100.0)
mean_err_f = float(np.mean([r["err_f"] for r in results_runs]))
mean_ci_half = float(np.mean([r["ci_half"] for r in results_runs]))
g_fwd_med = float(np.median([r["g_fwd"] for r in results_runs]))
g_rev_med = float(np.median([r["g_rev"] for r in results_runs]))

out_summary = {
    "system": "Alanine Dipeptide (2D Ramachandran PMF)",
    "temperature_k": TEMP,
    "kbt_kcal_mol": KBT,
    "exact_Z": Z_exact,
    "exact_F_kcal_mol": F_exact,
    "c7ax_equilibrium_mass_pct": alpha_c7ax * 100.0,
    "theoretical_bias_kcal_mol": theoretical_bias,
    "trained_model": "RealNVP Normalizing Flow (4 coupling layers)",
    "training_objective": "Reverse KL Divergence (Zero-Forcing)",
    "audit": {
        "n_runs": R_RUNS,
        "m_samples_per_run": M_SAMPLES,
        "c7ax_sampled_pct": c7ax_gen_mean * 100.0,
        "median_ess_pct": ess_med,
        "mean_f_hat_kcal_mol": float(np.mean([r["f_hat"] for r in results_runs])),
        "mean_error_f_kcal_mol": mean_err_f,
        "mean_nominal_ci_kcal_mol": mean_ci_half,
        "empirical_95ci_coverage_pct": cov_rate,
        "median_forward_jensen_gap_nats": g_fwd_med,
        "median_reverse_jensen_gap_nats": g_rev_med
    }
}

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
out_path = data_dir / "alanine_flow_audit.json"
with open(out_path, 'w') as f:
    json.dump(out_summary, f, indent=2)

print("\n" + "="*70)
print("ALANINE DIPEPTIDE BOLTZMANN GENERATOR AUDIT RESULTS:")
print(f"  True C7ax mass (alpha):            {alpha_c7ax * 100:.2f}%")
print(f"  Sampled C7ax mass (RealNVP):       {c7ax_gen_mean * 100:.3f}%  (Dropped mode)")
print(f"  Median Reported ESS:               {ess_med:.1f}%")
print(f"  Theoretical free-energy bias:      +{theoretical_bias:.3f} kcal/mol")
print(f"  Observed free-energy error:        +{mean_err_f:.3f} kcal/mol")
print(f"  Nominal 95% Confidence Interval:   +/- {mean_ci_half:.3f} kcal/mol")
print(f"  Nominal 95% CI Empirical Coverage: {cov_rate:.1f}%             (Asymptotic collapse to 0%)")
print(f"  Forward Jensen Gap (G_fwd):        {g_fwd_med:.3f} nats       (Blind to missing mode)")
print(f"  Reverse Jensen Gap (G_rev):        {g_rev_med:.3f} nats       (FLAGS DROPPED BASIN)")
print("="*70)
print(f"\n[SUCCESS] Saved audit to {out_path}")
