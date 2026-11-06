"""Shared figure style: theme A ("typewriter lab") with a one-accent rule.

Rules are in docs/figure_style.md. In short: white paper, Latin Modern Mono, four
validated hues with a fixed meaning in every figure (rust = VAN-MADE or the focal /
flawed series, cobalt = SA-TI or the correct reference, jade = PA, ochre = masked
diffusion), every other series in graded grays with its own marker, direct labels
over legends, a finding title with a provenance subtitle, bottom axis only, faint
dotted horizontal guides.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Surfaces and ink ------------------------------------------------------------
PAPER = "#ffffff"
INK = "#2a2723"
INK2 = "#6b655c"
GRID = "#e2dacb"
REF = INK2  # reference lines (targets, zero bias, theory guides): dashed
SURFACE = PAPER  # kept for older callers
TEXT, TEXT2 = INK, INK2

# Four validated hues (all pairs pass CVD and normal-vision checks on PAPER) -----
RUST, COBALT, JADE, OCHRE = "#c94f27", "#2f62b8", "#069a8e", "#d4960f"
GRAYS = ["#4a463f", "#8c857a", "#b3ab9e"]
SLOTS = [RUST, COBALT, JADE, OCHRE, *GRAYS]  # legacy order for generic series
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Sequential ramp (one hue, light to dark), cobalt on paper ----------------------
SEQ_STEPS = ["#d9dfe9", "#b1c1dd", "#86a0cf", "#5a7ec0", "#2f62b8", "#234a8c", "#183361"]
BLUE_RAMP = SEQ_STEPS  # legacy name
SEQ = LinearSegmentedColormap.from_list("seq_cobalt", [PAPER, *SEQ_STEPS])

CLASSICAL = ["pt", "sa_ti", "ais", "smc", "pa", "mf", "tap"]
LEARNED = ["van_made", "diffusion", "gflownet", "van_transformer"]
SAMPLERS = ["metropolis", "heatbath", "pt", "smc", "pa"]
LABEL = {
    "pt": "PT-TI",
    "sa_ti": "SA-TI",
    "ais": "AIS",
    "smc": "SMC",
    "pa": "PA",
    "mf": "Mean field",
    "tap": "TAP",
    "van_made": "VAN-MADE",
    "diffusion": "Masked diffusion",
    "gflownet": "GFlowNet",
    "van_transformer": "VAN-transformer",
    "metropolis": "Metropolis",
    "heatbath": "Heat-bath",
}

# Fixed identity: a method looks the same in every figure. -----------------------
METHOD_STYLE = {
    "van_made": {"color": RUST, "marker": "o"},
    "van_transformer": {"color": RUST, "marker": "o", "mfc": PAPER},
    "sa_ti": {"color": COBALT, "marker": "s"},
    "pa": {"color": JADE, "marker": "^"},
    "diffusion": {"color": OCHRE, "marker": "D"},
    "pt": {"color": GRAYS[1], "marker": "v"},
    "gflownet": {"color": GRAYS[0], "marker": "*"},
    "ais": {"color": GRAYS[1], "marker": "P"},
    "tap": {"color": GRAYS[2], "marker": "h"},
    "metropolis": {"color": GRAYS[1], "marker": "X"},
    "smc": {"color": GRAYS[2], "marker": "X"},
    "mf": {"color": GRAYS[2], "marker": "d"},
    "heatbath": {"color": GRAYS[2], "marker": "P"},
}


def style(group, method: str) -> dict:
    """Color, marker and label for a method (the group argument is kept for older
    callers; identity no longer depends on the panel)."""
    s = dict(METHOD_STYLE.get(method, {"color": GRAYS[1], "marker": "o"}))
    s["label"] = LABEL.get(method, method)
    s.setdefault("mec", PAPER)
    s.setdefault("mew", 0.7)
    return s


def titles(ax, main: str, sub: str | None = None):
    """Finding title (left) and provenance subtitle (data, sizes, interval type)."""
    ax.set_title(main, loc="left", pad=15 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=6.6, color=INK2, va="bottom")


def end_label(ax, x, y, text, dx=4, dy=0, **kw):
    """Direct label at a series end, in ink (never the series color)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        fontsize=kw.pop("fontsize", 6.8),
        color=kw.pop("color", INK),
        **kw,
    )


def setup():
    mpl.rcParams.update(
        {
            "font.family": ["Latin Modern Mono", "DejaVu Sans Mono"],
            "mathtext.fontset": "cm",
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.major.size": 3,
            "ytick.major.size": 0,
            "ytick.minor.size": 0,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": (0, (1, 2)),
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "errorbar.capsize": 0,
            "font.size": 8,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
        }
    )
    return plt


def spread_labels(ax, items, dx=6, min_gap=8.5, fontsize=6.6):
    """Direct labels for several series ends that may collide.

    items: list of (x, y, text) in data coordinates. Labels are placed to the right of
    their points and pushed apart vertically to at least min_gap points, with a thin
    leader line when a label moved. Text is in ink.
    """
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_disp = ax.transData.transform
    pts = [(to_disp((x, y)), x, y, t) for x, y, t in items]
    pts.sort(key=lambda p: p[0][1])
    scale = fig.dpi / 72.0
    ys = [p[0][1] for p in pts]
    gap = min_gap * scale
    for _ in range(50):  # relax: push neighbours apart symmetrically
        moved = False
        for i in range(1, len(ys)):
            d = ys[i] - ys[i - 1]
            if d < gap:
                shift = (gap - d) / 2
                ys[i - 1] -= shift
                ys[i] += shift
                moved = True
        if not moved:
            break
    for (disp, x, y, t), yd in zip(pts, ys, strict=True):
        off = (yd - disp[1]) / scale
        kw = {}
        if abs(off) > 1.5:
            kw["arrowprops"] = {
                "arrowstyle": "-",
                "color": INK2,
                "lw": 0.5,
                "shrinkA": 0,
                "shrinkB": 2,
            }
        ax.annotate(
            t,
            (x, y),
            xytext=(dx, off),
            textcoords="offset points",
            va="center",
            fontsize=fontsize,
            color=INK,
            annotation_clip=False,
            **kw,
        )


def plain_log_axis(ax, axis="y"):
    """Plain log tick labels: 0.01, 0.1, 1, 10 or 1e-4 for small values."""
    import math

    from matplotlib.ticker import FuncFormatter, NullFormatter

    def fmt(v, _):
        if v <= 0:
            return ""
        e = math.log10(v)
        if abs(e - round(e)) > 1e-9:
            return f"{v:g}"
        return f"{v:g}" if -2 <= e <= 3 else f"1e{int(round(e))}"

    a = getattr(ax, f"{axis}axis")
    a.set_major_formatter(FuncFormatter(fmt))
    a.set_minor_formatter(NullFormatter())
