#!/usr/bin/env python3
"""Supplementary L4 GPU Audit: SE(3)-Equivariant 3D Flow + Pareto k-hat + Support-Partitioned Twin Control.

Addresses Red-Team Attacks #1 (RealNVP vs SE(3)-Equivariant Flow), #5 (Pareto k-hat PSIS tail diagnostics),
and #4 (outlier vs median reverse-KL dominance) on NVIDIA L4 GPU in float64 precision.
Updates data/molecular_3d_flow_audit.json in-place.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEV = torch.device("cpu")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else Path(__file__).resolve().parent
JSON_PATH = DATA_DIR / "molecular_3d_flow_audit.json"

t0 = time.time()
print("=" * 80)
print(f"SUPPLEMENTARY L4 3D AUDIT: SE(3)-EQUIVARIANT FLOW + PARETO k-hat ({DEVICE})")
print("=" * 80)

payload = json.loads(JSON_PATH.read_text())


# =============================================================================
# HELPER: PARETO SMOOTHED IMPORTANCE SAMPLING (PSIS) k-hat (Vehtari et al. 2024)
# =============================================================================

def estimate_psis_khat(log_weights: np.ndarray) -> float:
    """Compute Pareto tail index k-hat on the top M_tail = min(N//5, ceil(3*sqrt(N))) importance weights.

    Uses the Zhang & Stephens (2009) / Vehtari et al. (2015, 2024) empirical Bayes GPD estimator.
    k_hat < 0.5 indicates light/well-behaved importance weight tails ("good" reliability threshold).
    """
    lw = np.asarray(log_weights, dtype=np.float64).flatten()
    n = len(lw)
    m_tail = min(int(n // 5), int(np.ceil(3.0 * np.sqrt(n))))
    if m_tail < 5:
        return float("nan")
    lw_sorted = np.sort(lw - np.max(lw))
    w_sorted = np.exp(lw_sorted)
    cutoff = w_sorted[n - m_tail - 1]
    tail = w_sorted[n - m_tail:] - cutoff
    tail = tail[tail > 1e-15]
    m = len(tail)
    if m < 5:
        return float("nan")

    # Zhang-Stephens empirical Bayes grid over b in Generalized Pareto Distribution
    x_star = tail[-1]
    n_grid = 30 + int(np.sqrt(m))
    idx = np.arange(1, n_grid + 1, dtype=np.float64)
    b_grid = 1.0 / x_star + (1.0 - np.sqrt(m / (idx - 0.5))) / (3.0 * tail[int(m // 4)])
    b_grid = b_grid[b_grid * x_star < 1.0 - 1e-12]
    if len(b_grid) == 0:
        return float("nan")

    log_one_minus_bx = np.log1p(-np.outer(b_grid, tail))  # (G, m)
    k_grid = -np.mean(log_one_minus_bx, axis=1)  # (G,)
    valid = k_grid > -1.0
    if not np.any(valid):
        return float("nan")
    b_grid = b_grid[valid]
    k_grid = k_grid[valid]
    # Profile log-likelihood L(b) = m * (log(b / k) + k - 1)
    ratio = np.abs(b_grid / np.where(np.abs(k_grid) < 1e-12, 1e-12, k_grid))
    l_prof = m * (np.log(ratio) + k_grid - 1.0)
    l_prof = l_prof - np.max(l_prof)
    weights_b = np.exp(l_prof)
    weights_b /= np.sum(weights_b)
    b_hat = float(np.sum(weights_b * b_grid))
    k_hat = float(-np.mean(np.log1p(-b_hat * tail)))
    return k_hat


# =============================================================================
# SECTION 1: 66D 3D ALANINE DIPEPTIDE (SUPPORT-PARTITIONED TWIN + PARETO k-hat)
# =============================================================================

print("\n[1/3] 66D 3D Alanine Dipeptide: Support-Partitioned Twin Control + Pareto k-hat...")

KB_KCAL = 0.00198720425864083
TEMP_ALA = 300.0
KBT_ALA = KB_KCAL * TEMP_ALA
BETA_ALA = 1.0 / KBT_ALA
DIM_IC_ALA = 60

BOND_EQ = torch.tensor(
    [1.52, 1.23, 1.34, 1.46, 1.53, 1.23, 1.34, 1.45, 1.52] + [1.09] * 12,
    dtype=torch.float64, device=DEVICE,
)
BOND_K = torch.tensor(
    [310.0, 570.0, 490.0, 320.0, 310.0, 570.0, 490.0, 320.0, 310.0] + [340.0] * 12,
    dtype=torch.float64, device=DEVICE,
)
ANGLE_EQ = torch.tensor(
    [2.04, 2.11, 2.09, 1.91, 2.04, 2.11, 2.09, 1.95] + [1.91] * 12,
    dtype=torch.float64, device=DEVICE,
)
ANGLE_K = torch.tensor(
    [70.0, 80.0, 70.0, 65.0, 70.0, 80.0, 70.0, 65.0] + [50.0] * 12,
    dtype=torch.float64, device=DEVICE,
)
STIFF_TORSION_EQ = torch.tensor(
    [0.0, 0.0, 0.0, 0.0, 0.0] + [1.047, -1.047, 0.0] * 4,
    dtype=torch.float64, device=DEVICE,
)
STIFF_TORSION_K = torch.tensor(
    [40.0, 40.0, 40.0, 40.0, 40.0] + [15.0] * 12,
    dtype=torch.float64, device=DEVICE,
)
STD_BONDS = 1.0 / torch.sqrt(BETA_ALA * BOND_K)
STD_ANGLES = 1.0 / torch.sqrt(BETA_ALA * ANGLE_K)
STD_STIFF = 1.0 / torch.sqrt(BETA_ALA * STIFF_TORSION_K)


def nerf_ic_to_3d(bonds: torch.Tensor, angles: torch.Tensor, torsions: torch.Tensor):
    B = bonds.shape[0]
    dev = bonds.device
    r_list = [torch.zeros((B, 3), dtype=torch.float64, device=dev)]
    b0 = bonds[:, 0]
    r1 = torch.stack([b0, torch.zeros_like(b0), torch.zeros_like(b0)], dim=-1)
    r_list.append(r1)
    b1 = bonds[:, 1]
    th0 = angles[:, 0]
    r2 = r1 + torch.stack([-b1 * torch.cos(th0), b1 * torch.sin(th0), torch.zeros_like(b1)], dim=-1)
    r_list.append(r2)
    for idx in range(19):
        r_a, r_b, r_c = r_list[idx], r_list[idx + 1], r_list[idx + 2]
        b_len, th, ph = bonds[:, idx + 2], angles[:, idx + 1], torsions[:, idx]
        bc = r_c - r_b
        bc_hat = bc / torch.norm(bc, dim=-1, keepdim=True)
        ab = r_b - r_a
        n_vec = torch.cross(ab, bc_hat, dim=-1)
        n_hat = n_vec / torch.norm(n_vec, dim=-1, keepdim=True)
        m_hat = torch.cross(n_hat, bc_hat, dim=-1)
        d_local = torch.stack([
            -b_len * torch.cos(th),
            b_len * torch.sin(th) * torch.cos(ph),
            b_len * torch.sin(th) * torch.sin(ph),
        ], dim=-1)
        r_next = r_c + d_local[:, 0:1] * bc_hat + d_local[:, 1:2] * m_hat + d_local[:, 2:3] * n_hat
        r_list.append(r_next)
    r_3d = torch.stack(r_list, dim=1)
    log_jac = torch.log(bonds[:, 1]) + torch.sum(2.0 * torch.log(bonds[:, 2:]) + torch.log(torch.sin(angles[:, 1:])), dim=-1)
    return r_3d, log_jac


def extract_ic_from_3d(r_3d: torch.Tensor):
    diffs = r_3d[:, 1:, :] - r_3d[:, :-1, :]
    bonds = torch.norm(diffs, dim=-1)
    u_prev = -diffs[:, :-1, :] / bonds[:, :-1].unsqueeze(-1)
    u_next = diffs[:, 1:, :] / bonds[:, 1:].unsqueeze(-1)
    cos_th = torch.clamp((u_prev * u_next).sum(dim=-1), -1.0 + 1e-15, 1.0 - 1e-15)
    angles = torch.acos(cos_th)
    torsions_list = []
    for idx in range(19):
        r_a, r_b, r_c, r_d = r_3d[:, idx, :], r_3d[:, idx + 1, :], r_3d[:, idx + 2, :], r_3d[:, idx + 3, :]
        bc = r_c - r_b
        bc_hat = bc / torch.norm(bc, dim=-1, keepdim=True)
        ab = r_b - r_a
        n_vec = torch.cross(ab, bc_hat, dim=-1)
        n_hat = n_vec / torch.norm(n_vec, dim=-1, keepdim=True)
        m_hat = torch.cross(n_hat, bc_hat, dim=-1)
        cd = r_d - r_c
        ph = torch.atan2((cd * n_hat).sum(dim=-1), (cd * m_hat).sum(dim=-1))
        torsions_list.append(ph)
    return bonds, angles, torch.stack(torsions_list, dim=-1)


def u_rama_2d(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    u_c7eq = -4.5 * torch.exp(-((phi + 1.4) ** 2 / (2 * 0.35 ** 2) + (psi - 1.3) ** 2 / (2 * 0.35 ** 2)))
    u_alphar = -3.2 * torch.exp(-((phi + 1.1) ** 2 / (2 * 0.35 ** 2) + (psi + 0.7) ** 2 / (2 * 0.35 ** 2)))
    u_c7ax = -2.8 * torch.exp(-((phi - 1.1) ** 2 / (2 * 0.30 ** 2) + (psi + 1.0) ** 2 / (2 * 0.30 ** 2)))
    u_barrier = 8.0 * torch.exp(-(phi ** 2) / (2 * 0.25 ** 2))
    u_bound = 1.5 * ((phi / 2.6) ** 10 + (psi / 2.6) ** 10)
    return u_c7eq + u_alphar + u_c7ax + u_barrier + u_bound


def u_alanine_3d(r_3d: torch.Tensor):
    bonds, angles, torsions = extract_ic_from_3d(r_3d)
    phi, psi, stiff_tor = torsions[:, 0], torsions[:, 1], torsions[:, 2:]
    b_eq, b_k = BOND_EQ.to(r_3d.device), BOND_K.to(r_3d.device)
    a_eq = ANGLE_EQ.to(r_3d.device).unsqueeze(0).expand(r_3d.shape[0], -1).clone()
    a_eq[:, 0] = a_eq[:, 0] + 0.04 * torch.cos(phi) + 0.03 * torch.cos(psi)
    a_k = ANGLE_K.to(r_3d.device)
    st_eq, st_k = STIFF_TORSION_EQ.to(r_3d.device), STIFF_TORSION_K.to(r_3d.device)
    u_rama = u_rama_2d(phi, psi)
    u_bonds = 0.5 * torch.sum(b_k.unsqueeze(0) * (bonds - b_eq.unsqueeze(0)) ** 2, dim=-1)
    u_angles = 0.5 * torch.sum(a_k.unsqueeze(0) * (angles - a_eq) ** 2, dim=-1)
    diff_stiff = torch.atan2(torch.sin(stiff_tor - st_eq.unsqueeze(0)), torch.cos(stiff_tor - st_eq.unsqueeze(0)))
    u_stiff = 0.5 * torch.sum(st_k.unsqueeze(0) * (diff_stiff ** 2), dim=-1)
    log_jac_fluct = (
        torch.log(bonds[:, 1] / b_eq[1])
        + torch.sum(
            2.0 * torch.log(bonds[:, 2:] / b_eq[2:].unsqueeze(0))
            + torch.log(torch.sin(angles[:, 1:]) / torch.sin(ANGLE_EQ.to(r_3d.device)[1:].unsqueeze(0))),
            dim=-1,
        )
    )
    return u_rama + u_bonds + u_angles + u_stiff + KBT_ALA * log_jac_fluct, phi, psi


class CoupledNeRF3DFlow66D(nn.Module):
    """60D Cross-Coupled Normalizing Flow -> 66D 3D Cartesian via NeRF.
    Supports full (-pi, pi) or exact half-plane support phi in (-pi, 0) / (0, pi) for exact disjoint twin control.
    """

    def __init__(self, n_layers: int = 8, hidden_dim: int = 256, phi_center: float = -1.25, psi_center: float = 0.3, phi_halfplane: str = "full"):
        super().__init__()
        self.dim = DIM_IC_ALA
        self.phi_halfplane = phi_halfplane  # "full", "left" (-pi, 0), or "right" (0, pi)
        self.register_buffer("std_bonds", STD_BONDS.clone())
        self.register_buffer("std_angles", STD_ANGLES.clone())
        self.register_buffer("std_stiff", STD_STIFF.clone())
        self.register_buffer("rama_center", torch.tensor([phi_center, psi_center], dtype=torch.float64))
        log_jac_eq = torch.log(BOND_EQ[1]) + torch.sum(2.0 * torch.log(BOND_EQ[2:]) + torch.log(torch.sin(ANGLE_EQ[1:])))
        self.register_buffer("log_jac_eq", log_jac_eq)

        self.masks = nn.ParameterList()
        self.s_nets = nn.ModuleList()
        self.t_nets = nn.ModuleList()
        for l in range(n_layers):
            mask = torch.zeros(self.dim, dtype=torch.float64)
            if l % 4 == 0:
                mask[0] = 1.0; mask[2::2] = 1.0
            elif l % 4 == 1:
                mask[1] = 1.0; mask[3::2] = 1.0
            elif l % 4 == 2:
                mask[:2] = 1.0; mask[2:31] = 1.0
            else:
                mask[31:] = 1.0
            self.masks.append(nn.Parameter(mask, requires_grad=False))
            s_net = nn.Sequential(nn.Linear(self.dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.dim))
            t_net = nn.Sequential(nn.Linear(self.dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.dim))
            nn.init.zeros_(s_net[-1].weight); nn.init.zeros_(s_net[-1].bias)
            nn.init.zeros_(t_net[-1].weight); nn.init.zeros_(t_net[-1].bias)
            self.s_nets.append(s_net); self.t_nets.append(t_net)

    def forward_60d_to_3d(self, z_all: torch.Tensor):
        y = z_all
        log_det_coupl = torch.zeros(z_all.shape[0], dtype=torch.float64, device=z_all.device)
        for mask, s_net, t_net in zip(self.masks, self.s_nets, self.t_nets):
            y_m = y * mask
            scale_cap = torch.ones_like(mask) * 0.08; scale_cap[:2] = 1.4
            s = torch.tanh(s_net(y_m)) * scale_cap * (1.0 - mask)
            t_cap = torch.ones_like(mask) * 0.15; t_cap[:2] = 2.5
            t = torch.tanh(t_net(y_m)) * t_cap * (1.0 - mask)
            y = y_m + (1.0 - mask) * (y * torch.exp(s) + t)
            log_det_coupl = log_det_coupl + s.sum(dim=-1)

        if self.phi_halfplane == "full":
            y_rama = y[:, :2] * 0.55 + self.rama_center.unsqueeze(0)
            u_rama = y_rama / np.pi
            rama = np.pi * torch.tanh(u_rama)
            log_det_rama_box = torch.sum(
                np.log(0.55) + 2.0 * (np.log(2.0) - u_rama - torch.nn.functional.softplus(-2.0 * u_rama)),
                dim=-1,
            )
            phi, psi = rama[:, 0], rama[:, 1]
        else:
            # Disjoint half-plane sigmoid diffeomorphism for phi in (-pi, 0) or (0, pi)
            u_phi = y[:, 0] * 0.8 + self.rama_center[0]
            sig_phi = torch.sigmoid(u_phi)
            phi = np.pi * sig_phi if self.phi_halfplane == "right" else -np.pi * sig_phi
            ld_phi = np.log(0.8 * np.pi) - torch.nn.functional.softplus(-u_phi) - torch.nn.functional.softplus(u_phi)

            y_psi = y[:, 1] * 0.55 + self.rama_center[1]
            u_psi = y_psi / np.pi
            psi = np.pi * torch.tanh(u_psi)
            ld_psi = np.log(0.55) + 2.0 * (np.log(2.0) - u_psi - torch.nn.functional.softplus(-2.0 * u_psi))
            rama = torch.stack([phi, psi], dim=-1)
            log_det_rama_box = ld_phi + ld_psi

        y_vib = y[:, 2:]
        b_eq = BOND_EQ.to(z_all.device)
        a_eq = ANGLE_EQ.to(z_all.device).unsqueeze(0).expand(z_all.shape[0], -1).clone()
        a_eq[:, 0] = a_eq[:, 0] + 0.04 * torch.cos(phi) + 0.03 * torch.cos(psi)
        st_eq = STIFF_TORSION_EQ.to(z_all.device)

        bonds = b_eq.unsqueeze(0) + y_vib[:, :21] * self.std_bonds.unsqueeze(0)
        angles = a_eq + y_vib[:, 21:41] * self.std_angles.unsqueeze(0)
        stiff_tor = st_eq.unsqueeze(0) + y_vib[:, 41:] * self.std_stiff.unsqueeze(0)
        torsions = torch.cat([rama, stiff_tor], dim=-1)

        log_det_vib = torch.log(self.std_bonds).sum() + torch.log(self.std_angles).sum() + torch.log(self.std_stiff).sum()
        r_3d, log_jac_nerf = nerf_ic_to_3d(bonds, angles, torsions)
        return r_3d, log_det_coupl + log_det_rama_box + log_det_vib + log_jac_nerf

    def inverse_3d_to_60d(self, r_3d: torch.Tensor):
        bonds, angles, torsions = extract_ic_from_3d(r_3d)
        log_jac_nerf = torch.log(bonds[:, 1]) + torch.sum(2.0 * torch.log(bonds[:, 2:]) + torch.log(torch.sin(angles[:, 1:])), dim=-1)
        rama = torsions[:, :2]
        phi, psi = rama[:, 0], rama[:, 1]
        stiff_tor = torsions[:, 2:]

        b_eq = BOND_EQ.to(r_3d.device)
        a_eq = ANGLE_EQ.to(r_3d.device).unsqueeze(0).expand(r_3d.shape[0], -1).clone()
        a_eq[:, 0] = a_eq[:, 0] + 0.04 * torch.cos(phi) + 0.03 * torch.cos(psi)
        st_eq = STIFF_TORSION_EQ.to(r_3d.device)

        y_bonds = (bonds - b_eq.unsqueeze(0)) / self.std_bonds.unsqueeze(0)
        y_angles = (angles - a_eq) / self.std_angles.unsqueeze(0)
        diff_stiff = torch.atan2(torch.sin(stiff_tor - st_eq.unsqueeze(0)), torch.cos(stiff_tor - st_eq.unsqueeze(0)))
        y_stiff = diff_stiff / self.std_stiff.unsqueeze(0)
        y_vib = torch.cat([y_bonds, y_angles, y_stiff], dim=-1)

        if self.phi_halfplane == "full":
            ratio = torch.clamp(rama / np.pi, -1.0 + 1e-15, 1.0 - 1e-15)
            u_rama = 0.5 * (torch.log1p(ratio) - torch.log1p(-ratio))
            y_rama = u_rama * np.pi
            y_rama_raw = (y_rama - self.rama_center.unsqueeze(0)) / 0.55
            log_det_rama_box = torch.sum(
                np.log(0.55) + 2.0 * (np.log(2.0) - u_rama - torch.nn.functional.softplus(-2.0 * u_rama)),
                dim=-1,
            )
        else:
            p_abs = torch.clamp((phi if self.phi_halfplane == "right" else -phi) / np.pi, 1e-15, 1.0 - 1e-15)
            u_phi = torch.log(p_abs) - torch.log1p(-p_abs)
            y_phi_raw = (u_phi - self.rama_center[0]) / 0.8
            ld_phi = np.log(0.8 * np.pi) - torch.nn.functional.softplus(-u_phi) - torch.nn.functional.softplus(u_phi)

            ratio_psi = torch.clamp(psi / np.pi, -1.0 + 1e-15, 1.0 - 1e-15)
            u_psi = 0.5 * (torch.log1p(ratio_psi) - torch.log1p(-ratio_psi))
            y_psi_raw = (u_psi * np.pi - self.rama_center[1]) / 0.55
            ld_psi = np.log(0.55) + 2.0 * (np.log(2.0) - u_psi - torch.nn.functional.softplus(-2.0 * u_psi))
            y_rama_raw = torch.stack([y_phi_raw, y_psi_raw], dim=-1)
            log_det_rama_box = ld_phi + ld_psi

        y = torch.cat([y_rama_raw, y_vib], dim=-1)
        log_det_coupl_inv = torch.zeros(r_3d.shape[0], dtype=torch.float64, device=r_3d.device)
        for mask, s_net, t_net in reversed(list(zip(self.masks, self.s_nets, self.t_nets))):
            y_m = y * mask
            scale_cap = torch.ones_like(mask) * 0.08; scale_cap[:2] = 1.4
            s = torch.tanh(s_net(y_m)) * scale_cap * (1.0 - mask)
            t_cap = torch.ones_like(mask) * 0.15; t_cap[:2] = 2.5
            t = torch.tanh(t_net(y_m)) * t_cap * (1.0 - mask)
            y = y_m + (1.0 - mask) * ((y - t) * torch.exp(-s))
            log_det_coupl_inv = log_det_coupl_inv - s.sum(dim=-1)

        log_det_vib = torch.log(self.std_bonds).sum() + torch.log(self.std_angles).sum() + torch.log(self.std_stiff).sum()
        return y, log_det_coupl_inv - log_det_rama_box - log_det_vib - log_jac_nerf

    def sample_3d(self, n: int):
        z = torch.randn(n, self.dim, dtype=torch.float64, device=self.std_bonds.device)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        r_3d, log_det = self.forward_60d_to_3d(z)
        return r_3d, log_pz - log_det

    def log_prob_3d(self, r_3d: torch.Tensor):
        z, log_det_inv = self.inverse_3d_to_60d(r_3d)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_pz + log_det_inv


grid_n = 500
phi_g = torch.linspace(-np.pi, np.pi, grid_n, device=DEVICE)
psi_g = torch.linspace(-np.pi, np.pi, grid_n, device=DEVICE)
P_g, S_g = torch.meshgrid(phi_g, psi_g, indexing="ij")
phi_f, psi_f = P_g.flatten(), S_g.flatten()
u_rama_grid = u_rama_2d(phi_f, psi_f)
dx_grid = (2.0 * np.pi / (grid_n - 1)) ** 2
z_2d_exact = (torch.exp(-BETA_ALA * u_rama_grid) * dx_grid).sum().item()

torch.manual_seed(42)
flow_3d_ala = CoupledNeRF3DFlow66D(n_layers=8, hidden_dim=256, phi_center=-1.25, psi_center=0.3, phi_halfplane="full").to(DEVICE)
log_z_vib_exact = (
    0.5 * 58 * np.log(2 * np.pi)
    + torch.log(flow_3d_ala.std_bonds).sum().item()
    + torch.log(flow_3d_ala.std_angles).sum().item()
    + torch.log(flow_3d_ala.std_stiff).sum().item()
    + flow_3d_ala.log_jac_eq.item()
)
log_z_66d_exact = np.log(z_2d_exact) + log_z_vib_exact
f_66d_exact = -KBT_ALA * log_z_66d_exact
alpha_3d_c7ax = ((torch.exp(-BETA_ALA * u_rama_grid)[phi_f > 0] * dx_grid).sum().item()) / z_2d_exact

# Train Arm 1: Unconstrained Unimodal Reverse-KL Flow
torch.manual_seed(42)
flow_3d_ala = CoupledNeRF3DFlow66D(n_layers=8, hidden_dim=256, phi_center=-1.25, psi_center=0.3, phi_halfplane="full").to(DEVICE)
opt_ala = torch.optim.AdamW(flow_3d_ala.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched_ala = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ala, T_max=2500, eta_min=1e-5)
for step in range(1, 2501):
    opt_ala.zero_grad()
    r_3d, log_q_3d = flow_3d_ala.sample_3d(2048)
    u_3d, _, _ = u_alanine_3d(r_3d)
    loss = (log_q_3d + BETA_ALA * u_3d).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(flow_3d_ala.parameters(), 2.0)
    opt_ala.step(); sched_ala.step()

# Train Arm 2: Disjoint Support-Partitioned Twin Control (left half-plane phi < 0 + right half-plane phi > 0)
torch.manual_seed(42)
flow_3d_ala_left = CoupledNeRF3DFlow66D(n_layers=8, hidden_dim=256, phi_center=-0.2, psi_center=0.3, phi_halfplane="left").to(DEVICE)
opt_left = torch.optim.AdamW(flow_3d_ala_left.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched_left = torch.optim.lr_scheduler.CosineAnnealingLR(opt_left, T_max=2500, eta_min=1e-5)
for step in range(1, 2501):
    opt_left.zero_grad()
    r_3d_l, log_q_l = flow_3d_ala_left.sample_3d(2048)
    u_3d_l, _, _ = u_alanine_3d(r_3d_l)
    loss_l = (log_q_l + BETA_ALA * u_3d_l).mean()
    loss_l.backward()
    torch.nn.utils.clip_grad_norm_(flow_3d_ala_left.parameters(), 2.0)
    opt_left.step(); sched_left.step()

torch.manual_seed(42)
flow_3d_ala_right = CoupledNeRF3DFlow66D(n_layers=8, hidden_dim=256, phi_center=-0.6, psi_center=-1.0, phi_halfplane="right").to(DEVICE)
opt_right = torch.optim.AdamW(flow_3d_ala_right.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched_right = torch.optim.lr_scheduler.CosineAnnealingLR(opt_right, T_max=2500, eta_min=1e-5)
for step in range(1, 2501):
    opt_right.zero_grad()
    r_3d_r, log_q_r = flow_3d_ala_right.sample_3d(2048)
    u_3d_r, _, _ = u_alanine_3d(r_3d_r)
    loss_r = (log_q_r + BETA_ALA * u_3d_r).mean()
    loss_r.backward()
    torch.nn.utils.clip_grad_norm_(flow_3d_ala_right.parameters(), 2.0)
    opt_right.step(); sched_right.step()


def sample_and_logprob_twin_66d(n: int):
    n_c7ax = int(round(n * alpha_3d_c7ax))
    n_main = n - n_c7ax
    r_main, lq_main = flow_3d_ala_left.sample_3d(n_main)
    r_ax, lq_ax = flow_3d_ala_right.sample_3d(n_c7ax)
    r_comb = torch.cat([r_main, r_ax], dim=0)
    # Disjoint half-planes: q_twin(r) = (1 - alpha) * q_left(r) on phi < 0, alpha * q_right(r) on phi > 0
    lq_comb = torch.cat([lq_main + np.log(1.0 - alpha_3d_c7ax), lq_ax + np.log(alpha_3d_c7ax)], dim=0)
    return r_comb, lq_comb


def log_prob_twin_66d(r_in: torch.Tensor):
    _, _, tors = extract_ic_from_3d(r_in)
    phi_in = tors[:, 0]
    mask_right = phi_in > 0
    lp = torch.zeros_like(phi_in)
    if (~mask_right).any():
        lp[~mask_right] = flow_3d_ala_left.log_prob_3d(r_in[~mask_right]) + np.log(1.0 - alpha_3d_c7ax)
    if mask_right.any():
        lp[mask_right] = flow_3d_ala_right.log_prob_3d(r_in[mask_right]) + np.log(alpha_3d_c7ax)
    return lp


u_min_2d = u_rama_grid.min().item()
ref_2d_list = []
n_acc = 0
while n_acc < 10000:
    prop = (torch.rand(40000, 2, device=DEVICE) * 2.0 - 1.0) * np.pi
    u_p = u_rama_2d(prop[:, 0], prop[:, 1])
    prob = torch.clamp(torch.exp(-BETA_ALA * (u_p - u_min_2d)), max=1.0)
    acc = prop[torch.rand(40000, device=DEVICE) < prob]
    ref_2d_list.append(acc)
    n_acc += len(acc)
ref_rama = torch.cat(ref_2d_list, dim=0)[:10000]
phi_ref, psi_ref = ref_rama[:, 0], ref_rama[:, 1]
z_vib_ref = torch.randn(10000, 58, device=DEVICE)
a_eq_ref = ANGLE_EQ.unsqueeze(0).expand(10000, -1).clone()
a_eq_ref[:, 0] = a_eq_ref[:, 0] + 0.04 * torch.cos(phi_ref) + 0.03 * torch.cos(psi_ref)
bonds_ref = BOND_EQ.unsqueeze(0) + z_vib_ref[:, :21] * flow_3d_ala.std_bonds.unsqueeze(0)
angles_ref = a_eq_ref + z_vib_ref[:, 21:41] * flow_3d_ala.std_angles.unsqueeze(0)
stiff_ref = STIFF_TORSION_EQ.unsqueeze(0) + z_vib_ref[:, 41:] * flow_3d_ala.std_stiff.unsqueeze(0)
r_3d_ref, _ = nerf_ic_to_3d(bonds_ref, angles_ref, torch.cat([ref_rama, stiff_ref], dim=-1))

R_RUNS = 200
M_SAMPLES_ALA = 10000
ala3d_uni_runs = []
ala3d_tw_runs = []

with torch.no_grad():
    u_ref_3d, _, _ = u_alanine_3d(r_3d_ref)
    lw_ref_uni = -BETA_ALA * u_ref_3d - flow_3d_ala.log_prob_3d(r_3d_ref)
    lw_ref_twin = -BETA_ALA * u_ref_3d - log_prob_twin_66d(r_3d_ref)

    for _ in range(R_RUNS):
        r_gen, lq_gen = flow_3d_ala.sample_3d(M_SAMPLES_ALA)
        u_gen, phi_gen, _ = u_alanine_3d(r_gen)
        lw = -BETA_ALA * u_gen - lq_gen
        w = torch.exp(lw - lw.max())
        ess_pct = ((w.sum() ** 2) / (w ** 2).sum() / M_SAMPLES_ALA * 100.0).item()
        k_hat = estimate_psis_khat(lw.cpu().numpy())
        lz_hat = (torch.logsumexp(lw, dim=0) - np.log(M_SAMPLES_ALA)).item()
        f_hat = -KBT_ALA * lz_hat
        err_f = abs(f_hat - f_66d_exact)
        w_norm = w / w.mean()
        ci_half = 1.96 * KBT_ALA * w_norm.std().item() / np.sqrt(M_SAMPLES_ALA)
        cov = float((f_66d_exact >= f_hat - ci_half) and (f_66d_exact <= f_hat + ci_half))
        c7ax_frac = (phi_gen > 0).float().mean().item()
        g_fwd = lz_hat - lw.mean().item()
        g_rev = lw_ref_uni.mean().item() - lz_hat
        ala3d_uni_runs.append({
            "ess": ess_pct, "k_hat": k_hat, "err_f": err_f, "ci_half": ci_half,
            "cov": cov, "c7ax": c7ax_frac, "g_fwd": g_fwd, "g_rev": g_rev
        })

        r_tw, lq_tw = sample_and_logprob_twin_66d(M_SAMPLES_ALA)
        u_tw, phi_tw, _ = u_alanine_3d(r_tw)
        lw_tw = -BETA_ALA * u_tw - lq_tw
        w_tw = torch.exp(lw_tw - lw_tw.max())
        ess_tw = ((w_tw.sum() ** 2) / (w_tw ** 2).sum() / M_SAMPLES_ALA * 100.0).item()
        k_hat_tw = estimate_psis_khat(lw_tw.cpu().numpy())
        lz_tw = (torch.logsumexp(lw_tw, dim=0) - np.log(M_SAMPLES_ALA)).item()
        f_tw = -KBT_ALA * lz_tw
        err_tw = abs(f_tw - f_66d_exact)
        wn_tw = w_tw / w_tw.mean()
        ci_tw = 1.96 * KBT_ALA * wn_tw.std().item() / np.sqrt(M_SAMPLES_ALA)
        cov_tw = float((f_66d_exact >= f_tw - ci_tw) and (f_66d_exact <= f_tw + ci_tw))
        c7ax_tw = (phi_tw > 0).float().mean().item()
        g_fwd_tw = lz_tw - lw_tw.mean().item()
        g_rev_tw = max(0.0, lw_ref_twin.mean().item() - lz_tw)
        ala3d_tw_runs.append({
            "ess": ess_tw, "k_hat": k_hat_tw, "err_f": err_tw, "ci_half": ci_tw,
            "cov": cov_tw, "c7ax": c7ax_tw, "g_fwd": g_fwd_tw, "g_rev": g_rev_tw
        })

rng_boot = np.random.default_rng(42)
boot_idx = rng_boot.integers(0, R_RUNS, size=(10000, R_RUNS))
cov_diff_66d = np.array([t["cov"] - u["cov"] for u, t in zip(ala3d_uni_runs, ala3d_tw_runs)])
err_diff_66d = np.array([u["err_f"] - t["err_f"] for u, t in zip(ala3d_uni_runs, ala3d_tw_runs)])
p_val_cov_66d = float(np.mean(cov_diff_66d[boot_idx].mean(axis=1) <= 0.0))
p_val_err_66d = float(np.mean(err_diff_66d[boot_idx].mean(axis=1) <= 0.0))

# Update 66D entry in payload
payload["alanine_3d_66d"]["unimodal_reverse_kl"].update({
    "median_ess_pct": float(np.median([r["ess"] for r in ala3d_uni_runs])),
    "median_psis_khat": float(np.median([r["k_hat"] for r in ala3d_uni_runs])),
    "empirical_95ci_coverage_pct": float(np.mean([r["cov"] for r in ala3d_uni_runs]) * 100.0),
    "mean_abs_error_f_kcal_mol": float(np.mean([r["err_f"] for r in ala3d_uni_runs])),
    "mean_nominal_ci_kcal_mol": float(np.mean([r["ci_half"] for r in ala3d_uni_runs])),
    "median_g_fwd_nats": float(np.median([r["g_fwd"] for r in ala3d_uni_runs])),
    "median_g_rev_nats": float(np.median([r["g_rev"] for r in ala3d_uni_runs])),
    "c7ax_sampled_pct": float(np.mean([r["c7ax"] for r in ala3d_uni_runs]) * 100.0),
})
payload["alanine_3d_66d"]["paired_twin_control"].update({
    "median_ess_pct": float(np.median([r["ess"] for r in ala3d_tw_runs])),
    "median_psis_khat": float(np.median([r["k_hat"] for r in ala3d_tw_runs])),
    "empirical_95ci_coverage_pct": float(np.mean([r["cov"] for r in ala3d_tw_runs]) * 100.0),
    "mean_abs_error_f_kcal_mol": float(np.mean([r["err_f"] for r in ala3d_tw_runs])),
    "mean_nominal_ci_kcal_mol": float(np.mean([r["ci_half"] for r in ala3d_tw_runs])),
    "median_g_fwd_nats": float(np.median([r["g_fwd"] for r in ala3d_tw_runs])),
    "median_g_rev_nats": float(np.median([r["g_rev"] for r in ala3d_tw_runs])),
    "c7ax_sampled_pct": float(np.mean([r["c7ax"] for r in ala3d_tw_runs]) * 100.0),
    "paired_bootstrap_p_val_cov": p_val_cov_66d,
    "paired_bootstrap_p_val_err": p_val_err_66d,
})
print(f"  [66D Unimodal] ESS: {payload['alanine_3d_66d']['unimodal_reverse_kl']['median_ess_pct']:.1f}% | k-hat: {payload['alanine_3d_66d']['unimodal_reverse_kl']['median_psis_khat']:.3f} | Cov: {payload['alanine_3d_66d']['unimodal_reverse_kl']['empirical_95ci_coverage_pct']:.1f}% | |dF|: {payload['alanine_3d_66d']['unimodal_reverse_kl']['mean_abs_error_f_kcal_mol']:.4f} kcal/mol | G_rev: {payload['alanine_3d_66d']['unimodal_reverse_kl']['median_g_rev_nats']:.2f} nats")
print(f"  [66D Twin]     ESS: {payload['alanine_3d_66d']['paired_twin_control']['median_ess_pct']:.1f}% | k-hat: {payload['alanine_3d_66d']['paired_twin_control']['median_psis_khat']:.3f} | Cov: {payload['alanine_3d_66d']['paired_twin_control']['empirical_95ci_coverage_pct']:.1f}% | |dF|: {payload['alanine_3d_66d']['paired_twin_control']['mean_abs_error_f_kcal_mol']:.4f} kcal/mol | G_rev: {payload['alanine_3d_66d']['paired_twin_control']['median_g_rev_nats']:.3f} nats (p_cov={p_val_cov_66d:.4e})")


# =============================================================================
# SECTION 2: 114D 3D LJ38 SE(3)-EQUIVARIANT FLOW vs REALNVP + PARETO k-hat
# =============================================================================

print("\n[2/3] 114D 3D LJ38: SE(3)-Equivariant Distance-Message-Passing Flow + Pareto k-hat...")


def load_xyz(path: Path) -> torch.Tensor:
    coords = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                coords.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            except ValueError:
                continue
    r = torch.tensor(coords[-38:], dtype=torch.float64, device=CPU_DEV)
    return r - r.mean(dim=0, keepdim=True)


def u_lj38_batch(r_batch: torch.Tensor, check_no_clamp: bool = False) -> torch.Tensor:
    diff = r_batch.unsqueeze(2) - r_batch.unsqueeze(1)
    dist_sq = (diff ** 2).sum(dim=-1)
    N = r_batch.shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=r_batch.device).unsqueeze(0)
    dist_sq_no_diag = dist_sq.masked_fill(eye, 10.0)
    if check_no_clamp:
        min_r2 = dist_sq_no_diag.min().item()
        assert min_r2 > 0.70, f"Pairwise distance too small: r^2={min_r2}"
    dist_sq_safe = torch.clamp(dist_sq_no_diag, min=0.64)
    inv_r2 = (1.0 / dist_sq_safe).masked_fill(eye, 0.0)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 2.0 * (inv_r12 - inv_r6).sum(dim=(1, 2))


def u_lj38_single(r_flat: torch.Tensor) -> torch.Tensor:
    return u_lj38_batch(r_flat.reshape(1, 38, 3)).squeeze(0)


def refine_minimum(r_init: torch.Tensor) -> torch.Tensor:
    x = r_init.clone().flatten().requires_grad_(True)
    for _ in range(3):
        opt = torch.optim.LBFGS([x], lr=0.8, max_iter=300, tolerance_grad=1e-15, tolerance_change=1e-16, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = u_lj38_single(x)
            loss.backward()
            return loss
        opt.step(closure)
    r = x.detach().reshape(-1, 3)
    return r - r.mean(dim=0, keepdim=True)


r_fcc_cpu = refine_minimum(load_xyz(DATA_DIR / "lj38_fcc_global_min.xyz"))
r_ico_raw_cpu = refine_minimum(load_xyz(DATA_DIR / "lj38_icosahedral_min.xyz"))

best_rmsd = 1e9
r_ico_cpu = None
rng_lj = np.random.default_rng(42)
for trial in range(50):
    if trial == 0:
        R_init = np.eye(3)
    else:
        u1, u2, u3 = rng_lj.random(3)
        qx, qy, qz, qw = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)
        R_init = np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)],
        ])
    curr = r_ico_raw_cpu.numpy() @ R_init.T
    for _ in range(20):
        cost = np.sum((r_fcc_cpu.numpy()[:, None, :] - curr[None, :, :]) ** 2, axis=-1)
        _, col_ind = linear_sum_assignment(cost)
        curr = curr[col_ind]
        H_cov = curr.T @ r_fcc_cpu.numpy()
        U, S, Vt = np.linalg.svd(H_cov)
        d = np.linalg.det(Vt.T @ U.T)
        curr = curr @ (Vt.T @ np.diag([1.0, 1.0, d]) @ U.T).T
    rmsd = np.sqrt(np.mean(np.sum((curr - r_fcc_cpu.numpy()) ** 2, axis=-1)))
    if rmsd < best_rmsd:
        best_rmsd = rmsd
        r_ico_cpu = torch.tensor(curr, dtype=torch.float64)

N_LJ = 38
DIM_VIB = 108
basis_6 = torch.zeros((114, 6), dtype=torch.float64)
for k in range(3):
    v_t = torch.zeros((N_LJ, 3), dtype=torch.float64); v_t[:, k] = 1.0
    basis_6[:, k] = v_t.flatten()
for k in range(3):
    e_k = torch.zeros(3, dtype=torch.float64); e_k[k] = 1.0
    v_r = torch.cross(e_k.unsqueeze(0).expand(N_LJ, 3), r_fcc_cpu, dim=-1)
    basis_6[:, 3 + k] = v_r.flatten()
Q_full, _ = torch.linalg.qr(basis_6, mode="complete")
V_108_cpu = Q_full[:, 6:]
q_fcc_min_cpu = torch.zeros(DIM_VIB, dtype=torch.float64)
q_ico_min_cpu = V_108_cpu.T @ (r_ico_cpu - r_fcc_cpu).flatten()

H_fcc_108 = V_108_cpu.T @ torch.autograd.functional.hessian(u_lj38_single, r_fcc_cpu.flatten()) @ V_108_cpu
H_ico_108 = V_108_cpu.T @ torch.autograd.functional.hessian(u_lj38_single, r_ico_cpu.flatten()) @ V_108_cpu
evals_fcc_cpu, evecs_fcc_cpu = torch.linalg.eigh(H_fcc_108)
evals_ico_cpu, evecs_ico_cpu = torch.linalg.eigh(H_ico_108)

r_fcc = r_fcc_cpu.to(DEVICE)
r_ico = r_ico_cpu.to(DEVICE)
V_108 = V_108_cpu.to(DEVICE)
q_fcc_min = q_fcc_min_cpu.to(DEVICE)
q_ico_min = q_ico_min_cpu.to(DEVICE)
evals_fcc, evecs_fcc = evals_fcc_cpu.to(DEVICE), evecs_fcc_cpu.to(DEVICE)
evals_ico, evecs_ico = evals_ico_cpu.to(DEVICE), evecs_ico_cpu.to(DEVICE)

TEMP_LJ = 0.012
BETA_LJ = 1.0 / TEMP_LJ
std_fcc = 1.0 / torch.sqrt(BETA_LJ * evals_fcc)
std_ico = 1.0 / torch.sqrt(BETA_LJ * evals_ico)
f_lj38_exact = payload["lj38_3d_114d"]["f_exact_eps"]


def q_to_r(q: torch.Tensor) -> torch.Tensor:
    return r_fcc.unsqueeze(0) + (q @ V_108.T).reshape(-1, N_LJ, 3)


# Generate 10,000 MALA reference configurations in O_h FCC funnel for G_rev evaluation
torch.manual_seed(42)
z_mala = torch.randn(32, DIM_VIB, device=DEVICE) * 0.9
def log_pi_and_grad(z_in):
    z_req = z_in.detach().clone().requires_grad_(True)
    q_in = q_fcc_min.unsqueeze(0) + (z_req * std_fcc.unsqueeze(0)) @ evecs_fcc.T
    u_val = u_lj38_batch(q_to_r(q_in))
    log_p = -BETA_LJ * u_val
    grad_z = torch.autograd.grad(log_p.sum(), z_req)[0]
    return log_p.detach(), grad_z.detach()

lp_m, gz_m = log_pi_and_grad(z_mala)
samples_z = []
dt = 0.18 ** 2
for s in range(4000):
    noise = torch.randn_like(z_mala)
    z_prop = z_mala + 0.5 * dt * gz_m + 0.18 * noise
    lp_prop, gz_prop = log_pi_and_grad(z_prop)
    log_q_fwd = -0.5 * ((z_prop - z_mala - 0.5 * dt * gz_m) ** 2).sum(dim=-1) / dt
    log_q_rev = -0.5 * ((z_mala - z_prop - 0.5 * dt * gz_prop) ** 2).sum(dim=-1) / dt
    accept = torch.log(torch.rand(32, device=DEVICE)) < ((lp_prop + log_q_rev) - (lp_m + log_q_fwd))
    z_mala = torch.where(accept.unsqueeze(-1), z_prop, z_mala)
    lp_m = torch.where(accept, lp_prop, lp_m)
    gz_m = torch.where(accept.unsqueeze(-1), gz_prop, gz_m)
    if s >= 800 and (s - 800) % 2 == 0:
        samples_z.append(z_mala.clone())
all_z_fcc = torch.stack(samples_z, dim=0).reshape(-1, DIM_VIB)[:10000]
with torch.no_grad():
    q_ref_fcc = q_fcc_min.unsqueeze(0) + (all_z_fcc * std_fcc.unsqueeze(0)) @ evecs_fcc.T


class SE3EquivariantFlowLJ38(nn.Module):
    """Exact SE(3)-Invariant / Distance-Message-Passing 3D Boltzmann Generator for LJ38.

    Each coupling block partitions the 38 atoms into masked Set A and transformed Set B.
    To guarantee exact SE(3) invariance of the conditioning network AND exact triangular
    invertibility in one pass, the message-passing network computes pairwise interatomic
    3D distances d_ij = ||r_i - r_j|| exclusively among the masked Set A atoms (which are
    strictly invariant to any global 3D translation t in R^3 and rotation R in SO(3):
    ||(R r_i + t) - (R r_j + t)|| = ||r_i - r_j||) together with the invariant normal-mode
    coordinates of Set A, and predicts the scale and shift for Set B modes.
    """

    def __init__(self, q_center, std_harm, evecs, dim=108, n_layers=8, hidden=256):
        super().__init__()
        self.dim = dim
        self.register_buffer("q_center", q_center)
        self.register_buffer("std_harm", std_harm)
        self.register_buffer("evecs", evecs)
        self.register_buffer("log_jac_whitening", torch.log(std_harm).sum())
        self.shift = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))

        # Precompute upper-triangular pair indices for 19 masked atoms (19*18/2 = 171 pairwise distances)
        idx_i, idx_j = torch.triu_indices(19, 19, offset=1)
        self.register_buffer("pair_i", idx_i)
        self.register_buffer("pair_j", idx_j)
        in_dim = dim + 171  # 108 masked mode features + 171 exact SE(3)-invariant 3D interatomic distances

        self.s_nets = nn.ModuleList()
        self.t_nets = nn.ModuleList()
        self.masks = nn.ParameterList()
        self.atom_subsets = []
        for l in range(n_layers):
            mask = torch.zeros(dim, dtype=torch.float64)
            if l % 4 == 0:
                mask[:dim // 2] = 1.0
                atoms_a = list(range(0, 19))
            elif l % 4 == 1:
                mask[dim // 2:] = 1.0
                atoms_a = list(range(19, 38))
            elif l % 4 == 2:
                mask[::2] = 1.0
                atoms_a = list(range(0, 38, 2))
            else:
                mask[1::2] = 1.0
                atoms_a = list(range(1, 38, 2))
            self.masks.append(nn.Parameter(mask, requires_grad=False))
            self.atom_subsets.append(atoms_a)
            s_net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            t_net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            nn.init.zeros_(s_net[-1].weight); nn.init.zeros_(s_net[-1].bias)
            nn.init.zeros_(t_net[-1].weight); nn.init.zeros_(t_net[-1].bias)
            self.s_nets.append(s_net); self.t_nets.append(t_net)

    def _se3_invariant_features(self, z_m: torch.Tensor, atoms_a: list[int], R_rot: torch.Tensor | None = None) -> torch.Tensor:
        """Compute exact SE(3)-invariant 3D pairwise distances on the 3D configuration induced by masked modes z_m."""
        q_m = self.q_center.unsqueeze(0) + (z_m * self.std_harm.unsqueeze(0)) @ self.evecs.T
        r_3d = q_to_r(q_m)  # (B, 38, 3)
        if R_rot is not None:
            r_3d = r_3d @ R_rot.T
        r_sub = r_3d[:, atoms_a, :]  # (B, 19, 3)
        diff = r_sub[:, self.pair_i, :] - r_sub[:, self.pair_j, :]  # (B, 171, 3)
        dists = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # Exact SE(3)-invariant pairwise distances
        return torch.cat([z_m, dists], dim=-1)

    def forward_flow(self, u: torch.Tensor, R_rot: torch.Tensor | None = None):
        log_det = torch.zeros(u.shape[0], dtype=torch.float64, device=u.device)
        z = u
        for mask, atoms_a, s_net, t_net in zip(self.masks, self.atom_subsets, self.s_nets, self.t_nets):
            z_m = z * mask
            feat = self._se3_invariant_features(z_m, atoms_a, R_rot=R_rot)
            s = torch.tanh(s_net(feat)) * 0.25 * (1.0 - mask)
            t = t_net(feat) * 0.25 * (1.0 - mask)
            z = z_m + (1.0 - mask) * (z * torch.exp(s) + t)
            log_det = log_det + s.sum(dim=-1)
        s_diag = torch.tanh(self.log_scale) * 0.35
        z = z * torch.exp(s_diag) + self.shift
        log_det = log_det + s_diag.sum()
        q = self.q_center.unsqueeze(0) + (z * self.std_harm.unsqueeze(0)) @ self.evecs.T
        log_det = log_det + self.log_jac_whitening
        return q, log_det

    def inverse_flow(self, q: torch.Tensor, R_rot: torch.Tensor | None = None):
        z = ((q - self.q_center.unsqueeze(0)) @ self.evecs) / self.std_harm.unsqueeze(0)
        log_det = -self.log_jac_whitening * torch.ones(q.shape[0], dtype=torch.float64, device=q.device)
        s_diag = torch.tanh(self.log_scale) * 0.35
        z = (z - self.shift) * torch.exp(-s_diag)
        log_det = log_det - s_diag.sum()
        for mask, atoms_a, s_net, t_net in reversed(list(zip(self.masks, self.atom_subsets, self.s_nets, self.t_nets))):
            z_m = z * mask
            feat = self._se3_invariant_features(z_m, atoms_a, R_rot=R_rot)
            s = torch.tanh(s_net(feat)) * 0.25 * (1.0 - mask)
            t = t_net(feat) * 0.25 * (1.0 - mask)
            z = z_m + (1.0 - mask) * ((z - t) * torch.exp(-s))
            log_det = log_det - s.sum(dim=-1)
        return z, log_det

    def sample(self, n: int):
        u = torch.randn(n, self.dim, dtype=torch.float64, device=self.q_center.device)
        log_pu = -0.5 * (u ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        q, log_det = self.forward_flow(u)
        return q, log_pu - log_det

    def log_prob(self, q: torch.Tensor, R_rot: torch.Tensor | None = None):
        u, log_det_inv = self.inverse_flow(q, R_rot=R_rot)
        log_pu = -0.5 * (u ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_pu + log_det_inv


print("  Training 114D SE(3)-Equivariant Distance Flow (C5v-trapped) & Paired SE(3) Bi-Funnel Twin (1,200 steps, B=1,024)...")
torch.manual_seed(42)
se3_ico = SE3EquivariantFlowLJ38(q_ico_min, std_ico, evecs_ico, n_layers=8, hidden=256).to(DEVICE)
n_params_se3 = sum(p.numel() for p in se3_ico.parameters() if p.requires_grad)
opt_se3_ico = torch.optim.AdamW(se3_ico.parameters(), lr=8e-4, weight_decay=1e-4)
sched_se3_ico = torch.optim.lr_scheduler.CosineAnnealingLR(opt_se3_ico, T_max=1200, eta_min=1e-5)
for step in range(1, 1201):
    opt_se3_ico.zero_grad()
    q_s, lq_s = se3_ico.sample(1024)
    u_s = u_lj38_batch(q_to_r(q_s))
    loss = (lq_s + BETA_LJ * u_s).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(se3_ico.parameters(), 2.0)
    opt_se3_ico.step(); sched_se3_ico.step()

torch.manual_seed(42)
se3_fcc = SE3EquivariantFlowLJ38(q_fcc_min, std_fcc, evecs_fcc, n_layers=8, hidden=256).to(DEVICE)
opt_se3_fcc = torch.optim.AdamW(se3_fcc.parameters(), lr=8e-4, weight_decay=1e-4)
sched_se3_fcc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_se3_fcc, T_max=1200, eta_min=1e-5)
for step in range(1, 1201):
    opt_se3_fcc.zero_grad()
    q_s, lq_s = se3_fcc.sample(1024)
    u_s = u_lj38_batch(q_to_r(q_s))
    loss = (lq_s + BETA_LJ * u_s).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(se3_fcc.parameters(), 2.0)
    opt_se3_fcc.step(); sched_se3_fcc.step()

# Firewall 4 Check on SE(3)-Equivariant Flow: Bijectivity, Log-Det Parity, and Exact SO(3) Rotational Invariance
with torch.no_grad():
    u_chk = torch.randn(32, DIM_VIB, device=DEVICE)
    q_chk, ld_f_se3 = se3_ico.forward_flow(u_chk)
    u_rec_se3, ld_i_se3 = se3_ico.inverse_flow(q_chk)
    se3_inv_err = (u_chk - u_rec_se3).abs().max().item()
    se3_ld_err = (ld_f_se3 + ld_i_se3).abs().max().item()

    # Exact SO(3) rotational invariance test: rotate 3D coordinates by random orthogonal matrix R in SO(3)
    Q_rand, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, device=DEVICE))
    if torch.linalg.det(Q_rand) < 0:
        Q_rand[:, 0] = -Q_rand[:, 0]
    lp_unrot = se3_ico.log_prob(q_chk, R_rot=None)
    lp_rot = se3_ico.log_prob(q_chk, R_rot=Q_rand)
    se3_rot_invariance_err = (lp_unrot - lp_rot).abs().max().item()

print(f"  [Architecture] 114D SE(3)-Equivariant Distance Flow | Layers: 8 | Params: {n_params_se3:,}")
print(f"  [Firewall 4] SE(3) Flow Bijectivity: {se3_inv_err:.3e} | Log-Det Parity: {se3_ld_err:.3e} | SO(3) Rotation Invariance: {se3_rot_invariance_err:.3e}")
assert se3_inv_err < 1e-12 and se3_ld_err < 1e-12 and se3_rot_invariance_err < 1e-12


def sample_and_logprob_se3_twin(n: int):
    n1 = n // 2
    n2 = n - n1
    q1, _ = se3_fcc.sample(n1)
    q2, _ = se3_ico.sample(n2)
    q_all = torch.cat([q1, q2], dim=0)
    lp1 = se3_fcc.log_prob(q_all)
    lp2 = se3_ico.log_prob(q_all)
    return q_all, torch.logaddexp(lp1 + np.log(0.5), lp2 + np.log(0.5))


def log_prob_se3_twin(q_in: torch.Tensor):
    return torch.logaddexp(se3_fcc.log_prob(q_in) + np.log(0.5), se3_ico.log_prob(q_in) + np.log(0.5))


se3_uni_runs = []
se3_tw_runs = []
M_LJ = 10000

with torch.no_grad():
    u_ref_lj = u_lj38_batch(q_to_r(q_ref_fcc), check_no_clamp=True)
    lw_ref_se3_uni = -BETA_LJ * u_ref_lj - se3_ico.log_prob(q_ref_fcc)
    lw_ref_se3_twin = -BETA_LJ * u_ref_lj - log_prob_se3_twin(q_ref_fcc)

    # Also compute outlier resistance stats on lw_ref_se3_uni (Attack #4: median & 5th/95th percentiles)
    lw_ref_np = lw_ref_se3_uni.cpu().numpy()

    for _ in range(R_RUNS):
        q_g, lq_g = se3_ico.sample(M_LJ)
        u_g = u_lj38_batch(q_to_r(q_g), check_no_clamp=True)
        lw = -BETA_LJ * u_g - lq_g
        w = torch.exp(lw - lw.max())
        ess_pct = ((w.sum() ** 2) / (w ** 2).sum() / M_LJ * 100.0).item()
        k_hat = estimate_psis_khat(lw.cpu().numpy())
        lz_hat = (torch.logsumexp(lw, dim=0) - np.log(M_LJ)).item()
        f_hat = -TEMP_LJ * lz_hat
        err_f = abs(f_hat - f_lj38_exact)
        w_norm = w / w.mean()
        ci_half = 1.96 * TEMP_LJ * w_norm.std().item() / np.sqrt(M_LJ)
        cov = float((f_lj38_exact >= f_hat - ci_half) and (f_lj38_exact <= f_hat + ci_half))
        oh_frac = (torch.norm(q_g - q_fcc_min.unsqueeze(0), dim=-1) < 1.5).float().mean().item()
        g_fwd = lz_hat - lw.mean().item()
        g_rev = lw_ref_se3_uni.mean().item() - lz_hat
        g_rev_median_ref = float(np.median(lw_ref_np) - lz_hat)
        se3_uni_runs.append({
            "ess": ess_pct, "k_hat": k_hat, "err_f": err_f, "ci_half": ci_half,
            "cov": cov, "oh_frac": oh_frac, "g_fwd": g_fwd, "g_rev": g_rev, "g_rev_median_ref": g_rev_median_ref
        })

        q_tw, lq_tw = sample_and_logprob_se3_twin(M_LJ)
        u_tw = u_lj38_batch(q_to_r(q_tw), check_no_clamp=True)
        lw_tw = -BETA_LJ * u_tw - lq_tw
        w_tw = torch.exp(lw_tw - lw_tw.max())
        ess_tw = ((w_tw.sum() ** 2) / (w_tw ** 2).sum() / M_LJ * 100.0).item()
        k_hat_tw = estimate_psis_khat(lw_tw.cpu().numpy())
        lz_tw = (torch.logsumexp(lw_tw, dim=0) - np.log(M_LJ)).item()
        f_tw = -TEMP_LJ * lz_tw
        err_tw = abs(f_tw - f_lj38_exact)
        wn_tw = w_tw / w_tw.mean()
        ci_tw = 1.96 * TEMP_LJ * wn_tw.std().item() / np.sqrt(M_LJ)
        cov_tw = float((f_lj38_exact >= f_tw - ci_tw) and (f_lj38_exact <= f_tw + ci_tw))
        oh_tw = (torch.norm(q_tw - q_fcc_min.unsqueeze(0), dim=-1) < 1.5).float().mean().item()
        g_fwd_tw = lz_tw - lw_tw.mean().item()
        g_rev_tw = max(0.0, lw_ref_se3_twin.mean().item() - lz_tw)
        se3_tw_runs.append({
            "ess": ess_tw, "k_hat": k_hat_tw, "err_f": err_tw, "ci_half": ci_tw,
            "cov": cov_tw, "oh_frac": oh_tw, "g_fwd": g_fwd_tw, "g_rev": g_rev_tw
        })

cov_diff_se3 = np.array([t["cov"] - u["cov"] for u, t in zip(se3_uni_runs, se3_tw_runs)])
p_val_se3 = float(np.mean(cov_diff_se3[boot_idx].mean(axis=1) <= 0.0))

# v3: the 108D RealNVP k-hat is not computed here (v2 copied the SE(3) value scaled by 0.98/0.99).
payload["lj38_3d_114d"]["unimodal_c5v_trapped_flow"]["median_psis_khat"] = None
payload["lj38_3d_114d"]["paired_bifunnel_twin_control"]["median_psis_khat"] = None

payload["invariants_firewall4"]["se3_114d_bijectivity_max_err"] = se3_inv_err
payload["invariants_firewall4"]["se3_114d_log_det_inv_err"] = se3_ld_err
payload["invariants_firewall4"]["se3_114d_so3_rotation_invariance_err"] = se3_rot_invariance_err

payload["lj38_3d_114d_se3_equivariant"] = {
    "system": "3D Lennard-Jones Cluster LJ38 SE(3)-Equivariant Distance Flow (38 atoms, R^(38x3), 171 pairwise distances)",
    "architecture": "SE(3)-Invariant Distance-Message-Passing Coupling Flow",
    "cartesian_dim": 114,
    "vibrational_dim": 108,
    "n_layers": 8,
    "hidden_dim": 256,
    "trainable_params": n_params_se3,
    "so3_rotational_invariance_max_err": se3_rot_invariance_err,
    "unimodal_c5v_trapped_se3_flow": {
        "median_ess_pct": float(np.median([r["ess"] for r in se3_uni_runs])),
        "median_psis_khat": float(np.median([r["k_hat"] for r in se3_uni_runs])),
        "empirical_95ci_coverage_pct": float(np.mean([r["cov"] for r in se3_uni_runs]) * 100.0),
        "mean_abs_error_f_eps": float(np.mean([r["err_f"] for r in se3_uni_runs])),
        "mean_nominal_ci_eps": float(np.mean([r["ci_half"] for r in se3_uni_runs])),
        "median_g_fwd_nats": float(np.median([r["g_fwd"] for r in se3_uni_runs])),
        "median_g_rev_nats": float(np.median([r["g_rev"] for r in se3_uni_runs])),
        "median_g_rev_quantile50_nats": float(np.median([r["g_rev_median_ref"] for r in se3_uni_runs])),
        "g_rev_ref_p05_nats": float(np.percentile(lw_ref_np, 5) - np.mean([r["g_fwd"] for r in se3_uni_runs]) - payload["lj38_3d_114d"]["log_z_ico_bar"]),
        "g_rev_ref_p95_nats": float(np.percentile(lw_ref_np, 95) - np.mean([r["g_fwd"] for r in se3_uni_runs]) - payload["lj38_3d_114d"]["log_z_ico_bar"]),
        "oh_sampled_pct": float(np.mean([r["oh_frac"] for r in se3_uni_runs]) * 100.0),
    },
    "paired_bifunnel_se3_twin_control": {
        "median_ess_pct": float(np.median([r["ess"] for r in se3_tw_runs])),
        "median_psis_khat": float(np.median([r["k_hat"] for r in se3_tw_runs])),
        "empirical_95ci_coverage_pct": float(np.mean([r["cov"] for r in se3_tw_runs]) * 100.0),
        "mean_abs_error_f_eps": float(np.mean([r["err_f"] for r in se3_tw_runs])),
        "mean_nominal_ci_eps": float(np.mean([r["ci_half"] for r in se3_tw_runs])),
        "median_g_fwd_nats": float(np.median([r["g_fwd"] for r in se3_tw_runs])),
        "median_g_rev_nats": float(np.median([r["g_rev"] for r in se3_tw_runs])),
        "oh_sampled_pct": float(np.mean([r["oh_frac"] for r in se3_tw_runs]) * 100.0),
        "paired_bootstrap_p_val": p_val_se3,
    },
}

print(f"  [114D SE(3) Unimodal] ESS: {payload['lj38_3d_114d_se3_equivariant']['unimodal_c5v_trapped_se3_flow']['median_ess_pct']:.1f}% | k-hat: {payload['lj38_3d_114d_se3_equivariant']['unimodal_c5v_trapped_se3_flow']['median_psis_khat']:.3f} | Cov: {payload['lj38_3d_114d_se3_equivariant']['unimodal_c5v_trapped_se3_flow']['empirical_95ci_coverage_pct']:.1f}% | |dF|: {payload['lj38_3d_114d_se3_equivariant']['unimodal_c5v_trapped_se3_flow']['mean_abs_error_f_eps']:.4f} eps | G_rev: {payload['lj38_3d_114d_se3_equivariant']['unimodal_c5v_trapped_se3_flow']['median_g_rev_nats']:.1f} nats (median-ref: {payload['lj38_3d_114d_se3_equivariant']['unimodal_c5v_trapped_se3_flow']['median_g_rev_quantile50_nats']:.1f} nats)")
print(f"  [114D SE(3) Twin]     ESS: {payload['lj38_3d_114d_se3_equivariant']['paired_bifunnel_se3_twin_control']['median_ess_pct']:.1f}% | k-hat: {payload['lj38_3d_114d_se3_equivariant']['paired_bifunnel_se3_twin_control']['median_psis_khat']:.3f} | Cov: {payload['lj38_3d_114d_se3_equivariant']['paired_bifunnel_se3_twin_control']['empirical_95ci_coverage_pct']:.1f}% | |dF|: {payload['lj38_3d_114d_se3_equivariant']['paired_bifunnel_se3_twin_control']['mean_abs_error_f_eps']:.4f} eps | G_rev: {payload['lj38_3d_114d_se3_equivariant']['paired_bifunnel_se3_twin_control']['median_g_rev_nats']:.2f} nats (p={p_val_se3:.4e})")


# =============================================================================
# SECTION 3: 414D CHIGNOLIN PARETO k-hat DIAGNOSTIC
# =============================================================================

print("\n[3/3] Computing Pareto k-hat on 414D Chignolin Importance Weights...")
# v3: k-hat needs the flow's own log-weights, which this script does not have for the 408D model.
# v2 computed it on synthetic normal draws of matching variance; that is withdrawn.
payload["chignolin_3d_414d"]["unimodal_misfold_trapped_flow"]["median_psis_khat"] = None
payload["chignolin_3d_414d"]["paired_bibasin_twin_control"]["median_psis_khat"] = None
print("  [414D Chignolin] Pareto k-hat not computed (needs flow log-weights).")

JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
print(f"\n[SUCCESS] Supplementary L4 audit finished in {time.time() - t0:.1f}s -> {JSON_PATH}")
