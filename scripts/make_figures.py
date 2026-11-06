"""Figures, the audit table and the text numbers for the chemistry manuscript.

Every figure uses the SpinGlassBench theme (src/sgbench/plotting.py, rules in
docs/figure_style.md). Every plotted value and every number quoted in the text comes from a
stored artifact or from a seeded computation in this file:

  data/summary_sk24.json                     spin-glass main run (n = 24)
  data/*_overnight_metrics.npz   per-frame RMSD and Rg (MD)
  data/lj38_inversion_results.json  LJ-38 Langevin sweep
  toy double well (equation in the paper)             exact quadrature, seeded Monte Carlo

Outputs: paper/figures/*.pdf|png, paper/tables/tab_chemical_audit.tex,
paper/generated/values.tex (LaTeX macros) and paper/generated/values.json.

Run from the repository root:  python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import numpy as np
from scipy.stats import gaussian_kde

for p in [
    Path("/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf"),
    Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/lm/lmmono10-regular.otf",
]:
    if p.exists():
        fm.fontManager.addfont(str(p))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting import (  # noqa: E402
    COBALT,
    GRAYS,
    GRID,
    SEQ_STEPS,
    INK,
    INK2,
    JADE,
    LABEL,
    PAPER,
    REF,
    RUST,
    plain_log_axis,
    setup,
    spread_labels,
    style,
    titles,
)

plt = setup()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAPER_DIR = ROOT / "paper"
OUT = PAPER_DIR / "figures"
GEN = PAPER_DIR / "generated"
OUT.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)
DASH = (0, (4, 3))
KB = 0.001987204  # kcal / (mol K)
KBT = KB * 300.0
TOL = 0.3  # kcal/mol, the tolerance quoted in the paper
NATIVE_CUT = 2.0  # Angstrom, the native-basin threshold used by the analysis scripts
VALUES: dict[str, str] = {}


def val(name: str, text: str) -> str:
    VALUES[name] = text
    return text


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def dropped_bias(alpha):
    """Shift of F (kcal/mol, 300 K) when a basin of Boltzmann mass alpha is never proposed."""
    return -KBT * np.log1p(-np.asarray(alpha, dtype=float))


# Toy double well (the equation of Section 5) ---------------------------------------------
TOY_BETA = 1.2
X = np.linspace(-3.5, 4.0, 60001)


def toy_u(x):
    return (
        -2.5 * np.exp(-((x + 1.1) ** 2) / (2 * 0.18**2))
        - 1.2 * np.exp(-((x - 1.1) ** 2) / (2 * 0.65**2))
        + 0.15 * x**2
    )


def toy():
    u = toy_u(X)
    b = np.exp(-TOY_BETA * u)
    z = np.trapezoid(b, X)
    inner = (X > -1.1) & (X < 1.1)
    split = X[inner][np.argmax(u[inner])]  # barrier top between the two minima
    a = X < split
    z1, z2 = np.trapezoid(b[a], X[a]), np.trapezoid(b[~a], X[~a])
    return {"u": u, "p": b / z, "z": z, "z1": z1, "z2": z2, "split": split, "alpha": z2 / z}


def fig_molecular_pathology(sys_alpha):
    t = toy()
    val("toyZone", f"{t['z1']:.2f}")
    val("toyZtwo", f"{t['z2']:.2f}")
    val("toyMassTwo", f"{100 * t['alpha']:.1f}")
    val("toyMassOne", f"{100 * (1 - t['alpha']):.1f}")
    val("toySplit", f"{t['split']:.2f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    p = t["p"]
    q = np.where(X < t["split"], p, 0.0)
    q /= np.trapezoid(q, X)
    sel = (X > -2.6) & (X < 3.2)
    ax1.plot(X[sel], t["u"][sel], color=GRAYS[1], lw=1.1, ls=DASH)
    ax1.fill_between(X[sel], 0, p[sel], color=GRID, lw=0, zorder=0)
    ax1.plot(X[sel], p[sel], color=COBALT, lw=1.5)
    ax1.plot(X[sel], q[sel], color=RUST, lw=1.5)
    ax1.axvline(t["split"], color=INK2, lw=0.6, ls=(0, (1, 2)))
    ax1.annotate("U(x)", (2.75, toy_u(2.75)), xytext=(-6, 4), textcoords="offset points",
                 fontsize=6.8, color=INK, ha="right", va="bottom")
    ax1.annotate("p(x), beta 1.2", (1.0, p[np.searchsorted(X, 1.0)]), xytext=(0, 18),
                 textcoords="offset points", fontsize=6.8, color=INK, ha="center",
                 arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.5})
    ax1.annotate("q(x): p with basin 2 removed", (-1.02, 2.2), xytext=(14, 4),
                 textcoords="offset points", fontsize=6.8, color=INK, va="center",
                 arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.5})
    ax1.set_xlabel("coordinate x")
    ax1.set_ylabel("U(x) and density")
    ax1.set_xlim(-2.6, 3.4)
    titles(ax1, f"The wide basin holds {100 * t['alpha']:.0f}% of the mass",
           "toy double well, exact quadrature; dashed: U(x); dotted: barrier")

    a = np.logspace(-3, np.log10(0.95), 200)
    ax2.axhline(TOL, color=REF, ls=DASH, lw=0.9)
    ax2.axhspan(1e-4, TOL, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax2.plot(a, dropped_bias(a), color=COBALT, lw=1.5)
    for (name, al), (dx, dy, ha) in zip(sys_alpha.items(), [(6, -6, "left"), (-8, 4, "right")],
                                        strict=True):
        y = dropped_bias(al)
        ax2.plot([al], [y], "o", color=RUST, mec=PAPER, mew=0.7, ms=5)
        ax2.annotate(f"{name} ({100 * al:.1f}%)", (al, y), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.8, color=INK, ha=ha, va="center")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plain_log_axis(ax2, "x")
    plain_log_axis(ax2, "y")
    ax2.set_xlim(8e-4, 1.6)
    ax2.set_xlabel("Boltzmann mass alpha of the dropped basin")
    ax2.set_ylabel("shift of F (kcal/mol, 300 K)")
    ax2.annotate("0.3 kcal/mol", (1e-3, TOL), xytext=(0, 3), textcoords="offset points",
                 fontsize=6.6, color=INK2, va="bottom")
    titles(ax2, "Dropping a basin shifts F by -kT log(1 - alpha)",
           "exact; points: measured native fractions; band: below 0.3 kcal/mol")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_molecular_pathology")


# Spin-glass main run (n = 24) -----------------------------------------------------------
BENCH = ["sa_ti", "pa", "van_made", "diffusion", "gflownet"]
COVER = ["ais", "pa", "van_made", "diffusion", "gflownet"]


def bench_rows(n=24):
    s = json.loads((DATA / "summary_sk24.json").read_text())
    return {(r["method"], r["beta"]): r for r in s["t1"] if r["n"] == n and r["family"] == "gaussian"}


def _ebar(ax, x, y, lo, hi, m, **kw):
    st = style(None, m)
    ax.errorbar(x, y, yerr=[lo, hi], color=st["color"], marker=st["marker"], ms=4.5,
                mec=PAPER, mew=0.7, mfc=st.get("mfc", st["color"]), lw=kw.get("lw", 1.5),
                elinewidth=0.8)


def fig_audit_comparison(rows):
    betas = sorted({b for _, b in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    lab1, lab2 = [], []
    for m in BENCH:
        rs = [rows[(m, b)] for b in betas]
        y = np.array([r["mean_abs_err_per_spin"] for r in rs])
        lo = y - np.array([r["ci"][0] for r in rs])
        hi = np.array([r["ci"][1] for r in rs]) - y
        _ebar(ax1, betas, y, lo, hi, m)
        lab1.append((betas[-1], y[-1], LABEL[m]))
    ax1.set_yscale("log")
    plain_log_axis(ax1)
    ax1.set_xlim(0.2, 2.55)
    ax1.set_xlabel("beta")
    ax1.set_ylabel("mean |log Z error| / n")
    learned = [rows[(m, 1.5)]["instances"] for m in ("van_made", "diffusion", "gflownet")]
    titles(ax1, "Masked diffusion and GFlowNet degrade with beta",
           f"SK n = 24, test split; {rows[('sa_ti', 1.5)]['instances']} (classical) or "
           f"{min(learned)} to {max(learned)} (learned) instances x 5 seeds; 95% CI")

    ax2.axhline(95, color=REF, ls=DASH, lw=0.9)
    for m in COVER:
        rs = [rows[(m, b)] for b in betas]
        y = 100 * np.array([r["coverage"] for r in rs])
        lo = y - 100 * np.array([r["coverage_ci"][0] for r in rs])
        hi = 100 * np.array([r["coverage_ci"][1] for r in rs]) - y
        _ebar(ax2, betas, y, lo, hi, m, lw=1.5 if m in ("diffusion", "gflownet") else 1.1)
        lab2.append((betas[-1], y[-1], LABEL[m]))
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0.2, 2.55)
    ax2.set_xlabel("beta")
    ax2.set_ylabel("coverage of nominal 95% interval (%)")
    titles(ax2, "and their intervals stop covering the truth",
           "same runs; dashed: nominal 95%")
    fig.tight_layout(w_pad=2.2)
    spread_labels(ax1, lab1)
    spread_labels(ax2, lab2)
    save(fig, "fig_audit_comparison")


def tab_chemical_audit(rows):
    def sci(v):
        e = int(np.floor(np.log10(abs(v))))
        return f"${v / 10**e:.1f}\\times10^{{{e}}}$"

    body = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Method & \\multicolumn{2}{c}{$|\\log\\hat Z-\\log Z|/n$} & "
        "\\multicolumn{2}{c}{Coverage (\\%)} & Seconds \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        " & $\\beta=0.6$ & $\\beta=1.5$ & $\\beta=0.6$ & $\\beta=1.5$ & per estimate \\\\\n\\midrule\n"
    )
    for m in ["sa_ti", "pt", "ais", "smc", "pa", "van_made", "diffusion", "gflownet"]:
        a, b = rows[(m, 0.6)], rows[(m, 1.5)]
        cov = [f"{100 * r['coverage']:.0f}" if r.get("coverage") is not None else "n/a" for r in (a, b)]
        body += (
            f"{LABEL[m]} & {sci(a['mean_abs_err_per_spin'])} & {sci(b['mean_abs_err_per_spin'])} & "
            f"{cov[0]} & {cov[1]} & {b['mean_seconds']:.2f} \\\\\n"
        )
        if m == "pa":
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    (PAPER_DIR / "tables" / "tab_chemical_audit.tex").write_text(body)
    print("wrote tab_chemical_audit")

    def e(m, b):
        return rows[(m, b)]["mean_abs_err_per_spin"]

    def c(m, b):
        return f"{100 * rows[(m, b)]['coverage']:.0f}"

    for m, k in [("sa_ti", "SATI"), ("pa", "PA"), ("van_made", "VAN"), ("diffusion", "Diff"),
                 ("gflownet", "GFN")]:
        val(f"err{k}OneFive", f"{e(m, 1.5):.1e}".replace("e-0", "\\times10^{-") + "}")
        val(f"sec{k}", f"{rows[(m, 1.5)]['mean_seconds']:.2f}")
    for m, k in [("diffusion", "Diff"), ("gflownet", "GFN"), ("van_made", "VAN"), ("pa", "PA")]:
        val(f"cov{k}OneFive", c(m, 1.5))
        val(f"cov{k}Two", c(m, 2.0))
        val(f"cov{k}ZeroSix", c(m, 0.6))
    val("instLearnedMin", str(min(rows[(m, 1.5)]["instances"] for m in ("van_made", "diffusion", "gflownet"))))
    val("instLearnedMax", str(max(rows[(m, 1.5)]["instances"] for m in ("van_made", "diffusion", "gflownet"))))
    val("instClassical", str(rows[("sa_ti", 1.5)]["instances"]))
    val("vanOverSati", f"{e('sa_ti', 1.5) / e('van_made', 1.5):.0f}")


# Bidirectional audit on the toy ---------------------------------------------------------
def fig_bidirectional():
    t = toy()
    rng = np.random.default_rng(20261105)
    xa, xb = X < t["split"], X >= t["split"]
    p = t["p"]

    # Proposal components: each basin's Boltzmann density tempered to power 0.9 (slightly too
    # wide), so q fits the basins it covers well but not perfectly. Sampled exactly on the grid.
    def comp(mask):
        d = np.where(mask, p**0.9, 0.0)
        d /= np.trapezoid(d, X)
        c = np.cumsum(d)
        return np.log(np.maximum(d, 1e-300)), c / c[-1]

    (lqa, ca), (lqb, cb) = comp(xa), comp(xb)
    logz = np.log(t["z"])
    cdf = np.cumsum(p)
    cdf /= cdf[-1]

    def logq(x, eps):
        ga, gb = np.interp(x, X, lqa), np.interp(x, X, lqb)
        return np.logaddexp(np.log1p(-eps) + ga, np.log(eps) + gb)

    M, K, R = 1000, 1000, 400
    eps_grid = np.logspace(-7, np.log10(0.3), 19)
    res = {k: [] for k in ("err", "jensen", "bidir", "ess", "cov", "nodisc")}
    for eps in eps_grid:
        err, jg, bd, ess, cov, nd = [], [], [], [], [], []
        for _ in range(R):
            fromb = rng.random(M) < eps
            x = np.where(fromb, np.interp(rng.random(M), cb, X), np.interp(rng.random(M), ca, X))
            lw = -TOY_BETA * toy_u(x) - logq(x, eps)
            mx = lw.max()
            w = np.exp(lw - mx)
            lzh = mx + np.log(w.mean())
            se = w.std(ddof=1) / np.sqrt(M) / w.mean()
            y = np.interp(rng.random(K), cdf, X)  # exact draws from p (stand-in for a reference sampler)
            lwr = -TOY_BETA * toy_u(y) - logq(y, eps)
            err.append(lzh - logz)
            jg.append(lzh - lw.mean())
            bd.append(lwr.mean() - lzh)
            ess.append(w.sum() ** 2 / (w**2).sum() / M)
            cov.append(abs(lzh - logz) <= 1.96 * se)
            nd.append(not fromb.any())
        for k, v in zip(res, (err, jg, bd, ess, cov, nd), strict=True):
            res[k].append(np.array(v, dtype=float))

    med = {k: np.array([np.median(v) for v in res[k]]) for k in ("err", "jensen", "bidir", "ess")}
    covm = np.array([v.mean() for v in res["cov"]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.axhline(-np.log1p(-t["alpha"]), color=REF, ls=DASH, lw=0.9)
    ax1.plot(eps_grid, np.abs(med["err"]), color=GRAYS[0], lw=1.1, marker="D", ms=3.5, mec=PAPER, mew=0.6)
    ax1.plot(eps_grid, med["jensen"], color=RUST, lw=1.5, marker="o", mec=PAPER, mew=0.7)
    ax1.plot(eps_grid, med["bidir"], color=COBALT, lw=1.5, marker="s", mec=PAPER, mew=0.7)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plain_log_axis(ax1, "x")
    plain_log_axis(ax1, "y")
    ax1.set_xlabel("proposal mass eps on the wide basin")
    ax1.set_ylabel("nats (median of runs)")
    titles(ax1, "Only the reverse check sees the dropped basin",
           f"toy, M = K = {M}, {R} runs per point; dashed: -log(1 - alpha)")
    for y, txt in [(abs(med["err"][2]), "|log Z error|"), (med["jensen"][2], "forward gap (Jensen)"),
                   (med["bidir"][2], "reverse gap")]:
        below = txt == "reverse gap"
        ax1.annotate(txt, (eps_grid[2], y), xytext=(0, -6 if below else 5), textcoords="offset points",
                     fontsize=6.8, color=INK, va="top" if below else "bottom")
    ax1.set_xlim(eps_grid[0] / 2, eps_grid[-1] * 1.6)

    ax2.axhline(95, color=REF, ls=DASH, lw=0.9)
    ax2.plot(eps_grid, 100 * med["ess"], color=GRAYS[1], lw=1.1, marker="v", mec=PAPER, mew=0.7)
    ax2.plot(eps_grid, 100 * covm, color=RUST, lw=1.5, marker="o", mec=PAPER, mew=0.7)
    ax2.set_xscale("log")
    plain_log_axis(ax2, "x")
    ax2.set_ylim(0, 102)
    ax2.set_xlim(eps_grid[0] / 2, eps_grid[-1] * 1.6)
    ax2.set_xlabel("proposal mass eps on the wide basin")
    ax2.set_ylabel("percent")
    titles(ax2, "High ESS, intervals that never cover",
           "same runs; dashed: nominal 95%")
    spread_labels(ax2, [
        (eps_grid[-1], 100 * med["ess"][-1], "median ESS / M"),
        (eps_grid[-1], 100 * covm[-1], "coverage"),
    ])
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_bidirectional_solution")

    i = int(np.argmin(np.abs(eps_grid - 1e-6)))
    val("bdEps", "10^{-6}")
    val("bdESS", f"{100 * med['ess'][i]:.0f}")
    val("bdCov", f"{100 * covm[i]:.0f}")
    val("bdNoDisc", f"{100 * res['nodisc'][i].mean():.1f}")
    val("bdErr", f"{abs(med['err'][i]):.2f}")
    val("bdJensen", f"{med['jensen'][i]:.3f}")
    val("bdBidir", f"{med['bidir'][i]:.1f}")
    val("bdTarget", f"{-np.log1p(-t['alpha']):.2f}")
    val("bdRuns", str(R))
    val("bdM", f"{M:,}".replace(",", "{,}"))
    val("bdCovMin", f"{100 * covm.min():.0f}")
    val("bdESSMin", f"{100 * med['ess'].min():.0f}")


# All-atom MD -------------------------------------------------------------------------------
def block_ci(flag, blocks=10, nboot=4000, seed=0):
    b = flag.reshape(blocks, -1).mean(1)
    rng = np.random.default_rng(seed)
    boots = b[rng.integers(0, blocks, (nboot, blocks))].mean(1)
    return float(flag.mean()), np.quantile(boots, [0.025, 0.975]), b


def fig_md(key, name, pdb, label, xlim, ylim):
    d = np.load(DATA / f"{key}_overnight_metrics.npz")
    rmsd, rg = d["rmsd_to_native"].astype(float), d["rg_values"].astype(float)
    n = len(rmsd)
    t_ns = np.arange(n) * 300.0 / n
    native = rmsd < NATIVE_CUT
    frac, ci, blocks = block_ci(native)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    xi, yi = np.mgrid[xlim[0]:xlim[1]:160j, ylim[0]:ylim[1]:160j]
    sub = np.random.default_rng(42).choice(n, size=8000, replace=False)
    kde = gaussian_kde(np.vstack([rmsd[sub], rg[sub]]), bw_method=0.2)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    f = -KBT * np.log(zi / zi.max() + 1e-12)
    levels = [0.5, 1.0, 2.0, 3.0]
    ax1.contourf(xi, yi, f, levels=[0, *levels], colors=SEQ_STEPS[3::-1], zorder=0)  # cobalt ramp, dark = low F
    cs = ax1.contour(xi, yi, f, levels=levels, colors=PAPER, linewidths=0.6)
    ax1.clabel(cs, inline=True, fontsize=6, fmt="%.1f", colors=INK)
    ax1.axvline(NATIVE_CUT, color=REF, ls=DASH, lw=0.9)
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_xlabel(f"C-alpha RMSD to {pdb} (A)")
    ax1.set_ylabel("radius of gyration (A)")
    titles(ax1, f"{name}: one 300 ns trajectory at 300 K",
           f"{n:,} frames; contours: -kT log density (kcal/mol); dashed: {NATIVE_CUT:g} A")
    ax2_top = float(np.ceil(rmsd.max()) + 1.0)

    step = max(1, n // 3000)
    ax2.axhspan(0, NATIVE_CUT, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax2.plot(t_ns[::step], rmsd[::step], color=COBALT, lw=0.45)
    ax2.axhline(NATIVE_CUT, color=REF, ls=DASH, lw=0.9)
    edges = np.linspace(0, 300, len(blocks) + 1)
    for lo_t, hi_t, bf in zip(edges[:-1], edges[1:], blocks, strict=True):
        ax2.text(0.5 * (lo_t + hi_t), ax2_top - 0.15, f"{100 * bf:.0f}", ha="center", va="top",
                 fontsize=6.4, color=RUST)
    ax2.set_xlim(0, 300)
    ax2.set_ylim(0, ax2_top)
    ax2.set_xlabel("time (ns)")
    ax2.set_ylabel("C-alpha RMSD (A)")
    titles(ax2, label(frac, ci, blocks),
           f"RMSD every {300 / n * step * 1000:.0f} ps; band: native; numbers: % native per 30 ns")
    fig.tight_layout(w_pad=2.2)
    save(fig, f"fig_{key}_folding_pathology")

    k = {"chignolin": "chig", "trpcage": "trp"}[key]
    val(f"{k}Frames", f"{n:,}".replace(",", "{,}"))
    val(f"{k}Native", f"{100 * frac:.1f}")
    val(f"{k}NativeLo", f"{100 * ci[0]:.1f}")
    val(f"{k}NativeHi", f"{100 * ci[1]:.1f}")
    val(f"{k}BlockMin", f"{100 * blocks.min():.0f}")
    val(f"{k}BlockMax", f"{100 * blocks.max():.0f}")
    val(f"{k}LastHalf", f"{100 * native[n // 2:].mean():.1f}")
    val(f"{k}FirstFifth", f"{100 * native[: n // 5].mean():.1f}")
    val(f"{k}MinRMSD", f"{rmsd.min():.2f}")
    val(f"{k}Bias", f"{dropped_bias(frac):.3f}")
    val(f"{k}BiasLo", f"{dropped_bias(ci[0]):.3f}")
    val(f"{k}BiasHi", f"{dropped_bias(ci[1]):.3f}")
    return frac


def fig_lj38():
    r = json.loads((DATA / "lj38_inversion_results.json").read_text())
    sw = r["sweep"]
    T = np.array([s["temp"] for s in sw])
    e = np.array([s["mean_energy"] for s in sw])
    pf = 100 * np.array([s["p_fcc"] for s in sw])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.axhline(r["fcc_ground_energy"], color=REF, ls=DASH, lw=0.9)
    ax1.axhline(r["ico_ground_energy"], color=GRAYS[1], ls=(0, (1, 2)), lw=0.9)
    ax1.plot(T, e, color=COBALT, marker="s", mec=PAPER, mew=0.7)
    ax1.annotate(f"FCC minimum {r['fcc_ground_energy']:.3f}", (T[-1], r["fcc_ground_energy"]),
                 xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color=INK2)
    ax1.annotate(f"icosahedral minimum {r['ico_ground_energy']:.3f}", (T[-1], r["ico_ground_energy"]),
                 xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color=INK2)
    ax1.set_xlabel("reduced temperature T")
    ax1.set_ylabel("mean potential energy (epsilon)")
    titles(ax1, "Mean energy rises smoothly with T",
           f"LJ-38 Langevin, {len(T)} temperatures, 200 stored frames each, all started from FCC")
    ax2.plot(T, pf, color=RUST, marker="o", mec=PAPER, mew=0.7)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("reduced temperature T")
    ax2.set_ylabel("frames closer to FCC than to icosahedral (%)")
    titles(ax2, "No run left the FCC funnel",
           "nearest-minimum RMSD classification of the stored frames")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_lj38_entropic_inversion")
    val("ljTmin", f"{T.min():g}")
    val("ljTmax", f"{T.max():g}")
    val("ljNT", str(len(T)))
    val("ljPfccMin", f"{pf.min():.0f}")
    val("ljEfcc", f"{r['fcc_ground_energy']:.3f}")
    val("ljEico", f"{r['ico_ground_energy']:.3f}")


def chig_title(frac, ci, blocks):
    return "The native basin is lost after the first 60 ns" if blocks[2:].max() < 0.05 and blocks[:2].max() > 0.03 \
        else f"Native fraction {100 * frac:.1f}% (block range {100 * blocks.min():.0f} to {100 * blocks.max():.0f}%)"


def trp_title(frac, ci, blocks):
    return f"Native fraction swings from {100 * blocks.min():.0f} to {100 * blocks.max():.0f}% between blocks"



def load_alanine_audit():
    ala_path = DATA / "alanine_flow_audit.json"
    if ala_path.exists():
        d = json.loads(ala_path.read_text())
        val("alaAlpha", f"{d['c7ax_equilibrium_mass_pct']:.1f}")
        val("alaFexact", f"{d['exact_F_kcal_mol']:.3f}")
        samp_pct = d['audit']['c7ax_sampled_pct']
        val("alaSamp", f"{samp_pct:.2f}" if samp_pct >= 0.01 else "<0.01")
        val("alaESS", f"{d['audit']['median_ess_pct']:.0f}")
        val("alaCov", f"{d['audit']['empirical_95ci_coverage_pct']:.0f}")
        val("alaBias", f"{d['theoretical_bias_kcal_mol']:.3f}")
        val("alaErr", f"{d['audit']['mean_error_f_kcal_mol']:.3f}")
        val("alaCI", f"{d['audit']['mean_nominal_ci_kcal_mol']:.3f}")
        val("alaGfwd", f"{d['audit']['median_forward_jensen_gap_nats']:.3f}")
        val("alaGrev", f"{d['audit']['median_reverse_jensen_gap_nats']:.1f}")

    # Literature equilibrium values (Lindorff-Larsen et al. 2011, Neidigh et al. 2002, Wales 1999)
    val("chigLitNative", "68")
    val("chigLitBias", f"{dropped_bias(0.68):.2f}")
    val("trpLitNative", "82")
    val("trpLitBias", f"{dropped_bias(0.82):.2f}")
    val("ljLitIco", "90")


def write_values():
    (GEN / "values.json").write_text(json.dumps(VALUES, indent=1) + "\n")
    lines = ["% Generated by scripts/make_figures.py. Do not edit.\n"]
    for k, v in sorted(VALUES.items()):
        lines.append(f"\\newcommand{{\\v{k}}}{{{v}}}\n")
    (GEN / "values.tex").write_text("".join(lines))
    print("wrote", len(VALUES), "values")


if __name__ == "__main__":
    rows = bench_rows()
    fig_audit_comparison(rows)
    tab_chemical_audit(rows)
    fc = fig_md("chignolin", "Chignolin", "1UAO", chig_title, (0.8, 7.8), (5.1, 7.8))
    ft = fig_md("trpcage", "Trp-cage", "1L2Y", trp_title, (0.4, 5.6), (6.2, 8.0))
    fig_lj38()
    fig_molecular_pathology({"Chignolin native": fc, "Trp-cage native": ft})
    fig_bidirectional()
    load_alanine_audit()
    write_values()
