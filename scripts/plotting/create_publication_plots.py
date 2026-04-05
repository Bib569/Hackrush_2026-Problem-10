#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Quality Benchmark Plots for Si & Ge Interatomic Potentials
======================================================================
Generates all benchmark figures in publication-ready format:
  - Serif fonts (STIX math), no titles (captions in LaTeX)
  - Bold axis labels, thick spines (2.5 pt), 300 DPI
  - PDF + PNG output
  - All 6 models included, layout issues fixed

Figures produced:
  1. fig1_lattice_parity        — Lattice constant parity (pred vs DFT)
  2. fig2_energy_parity         — Cohesive energy parity (pred vs DFT)
  3. fig3_eos_curves            — Equation of State (1×2: Si, Ge)
  4. fig4_strain_curves         — Energy–strain curves (1×2: Si, Ge)
  5. fig5_lattice_comparison    — Lattice constant grouped bar chart
  6. fig6_force_stability       — Force stability grouped bar chart
  7. fig7_per_model_lattice     — Per-model lattice parity (2×3)
  8. fig8_combined_correlation  — Combined 1×3 correlation panels
  9. fig9_optuna_convergence    — DeepMD Optuna HPO convergence
"""

import warnings
warnings.filterwarnings('ignore')

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================================
# GLOBAL STYLE  (matches user's VIF example exactly)
# ============================================================================
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

DPI = 300

def style_ax(ax, grid=False):
    """Apply publication styling to an axis: thick closed box, bold ticks."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_visible(True)
    ax.tick_params(which='major', width=2.5, length=7, labelsize=11,
                   direction='in', top=True, right=True)
    ax.tick_params(which='minor', width=1.5, length=4,
                   direction='in', top=True, right=True)
    if grid:
        ax.grid(True, alpha=0.20, linewidth=0.6, linestyle='-')
    else:
        ax.grid(False)

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Paths (WSL) ---
DATA_PRI = "/home/bib569/Workspace/Hackrush_2026-Problem-10/results_comprehensive"
DATA_SEC = "/mnt/c/Users/bibha/.gemini/antigravity/scratch"
OUT_DIR  = "/home/bib569/Workspace/Hackrush_2026-Problem-10/publication_plots"

# --- Models (canonical key → display, short) ---
#   Canonical key is what we normalize to internally.
MODELS = [
    ("MACE-MP-0",        "MACE-MP-0",             "MACE"),
    ("CHGNet",           "CHGNet",                 "CHGNet"),
    ("DPA-3-pretrained", "DPA-3 (SemiCond)",       "DPA-3"),
    ("DeepMD-finetuned", "DeepMD (fine-tuned)",     "DeepMD"),
    ("Tersoff",          "Tersoff",                "Tersoff"),
    ("SW",               "Stillinger-Weber",       "SW"),
]

# Key aliases: raw key in JSON → canonical key
KEY_ALIASES = {
    "MACE-MP-0":              "MACE-MP-0",
    "MACE-MP-0 (pre-trained)": "MACE-MP-0",
    "CHGNet":                 "CHGNet",
    "CHGNet (pre-trained)":    "CHGNet",
    "DPA-3-pretrained":       "DPA-3-pretrained",
    "DPA-3 (pre-trained)":    "DPA-3-pretrained",
    "DeepMD-finetuned":       "DeepMD-finetuned",
    "Tersoff":                "Tersoff",
    "SW":                     "SW",
    # We skip MEAM (empty) and Tersoff-Ge (duplicate)
}

# Deep, high-contrast palette for publication
COLORS = {
    'MACE-MP-0':        '#c41e3a',   # deep crimson
    'CHGNet':           '#1a5276',   # deep teal-blue
    'DPA-3-pretrained': '#6c3483',   # deep purple
    'DeepMD-finetuned': '#d35400',   # deep orange
    'Tersoff':          '#196f3d',   # deep forest green
    'SW':               '#2c3e50',   # dark slate
}

MARKERS = {
    'MACE-MP-0':        'o',
    'CHGNet':           's',
    'DPA-3-pretrained': 'D',
    'DeepMD-finetuned': '^',
    'Tersoff':          'v',
    'SW':               'P',
}

# Ground truth
GT = {
    "Si": {"dft_a": 5.469, "exp_a": 5.431, "dft_e": -5.43, "exp_e": -4.63},
    "Ge": {"dft_a": 5.763, "exp_a": 5.658, "dft_e": -4.62, "exp_e": -3.85},
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_json(fname):
    """Load JSON from primary or secondary data directory."""
    for d in [DATA_PRI, DATA_SEC]:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"Cannot find {fname} in {DATA_PRI} or {DATA_SEC}")


def normalize_keys(raw_data):
    """Map raw JSON keys to canonical keys used by MODELS list."""
    out = {}
    for raw_key, val in raw_data.items():
        canon = KEY_ALIASES.get(raw_key)
        if canon is None:
            continue  # skip unknown keys (MEAM empty, Tersoff-Ge duplicate)
        if not isinstance(val, dict) or not val:
            continue  # skip empty entries
        if canon in out:
            # merge (e.g. if both files contribute elements)
            out[canon].update(val)
        else:
            out[canon] = val
    return out


def load_data():
    """Load all benchmark data, merging primary and secondary sources."""
    merged = {}
    # Load from both sources and merge
    for d in [DATA_PRI, DATA_SEC]:
        p = os.path.join(d, "all_results_merged.json")
        if os.path.exists(p):
            with open(p) as f:
                raw = json.load(f)
            normed = normalize_keys(raw)
            for k, v in normed.items():
                if k in merged:
                    merged[k].update(v)
                else:
                    merged[k] = v

    if not merged:
        raise FileNotFoundError("Cannot find all_results_merged.json")

    optuna = None
    try:
        optuna = load_json("optuna_results.json")
    except FileNotFoundError:
        print("  ⚠  optuna_results.json not found — skipping convergence plot")
    return merged, optuna


# ============================================================================
# HELPER
# ============================================================================

def save_fig(fig, name):
    """Save figure as PDF + PNG."""
    for ext in ['pdf', 'png']:
        p = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    print(f"  ✓ {name}.pdf / .png")


def legend_pub(ax, **kw):
    """Create a publication-style legend."""
    defaults = dict(fontsize=10, framealpha=0.95, edgecolor='black',
                    fancybox=False, shadow=False)
    defaults.update(kw)
    return ax.legend(**defaults)


# ============================================================================
# FIGURE 1 — Lattice Constant Parity
# ============================================================================

def fig1_lattice_parity(data):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')
    style_ax(ax)

    xs_all, ys_all = [], []

    for mk, disp, short in MODELS:
        if mk not in data:
            continue
        for el in ["Si", "Ge"]:
            d = data[mk].get(el, {})
            if "a_eos" not in d:
                continue
            x = GT[el]["dft_a"]
            y = d["a_eos"]
            xs_all.append(x); ys_all.append(y)
            fc = COLORS[mk] if el == "Si" else "white"
            ax.scatter(x, y, marker=MARKERS[mk], s=140,
                       facecolors=fc, edgecolors=COLORS[mk],
                       linewidth=1.8, zorder=5)

    # parity line
    vmin = min(xs_all + ys_all) - 0.06
    vmax = max(xs_all + ys_all) + 0.06
    ax.plot([vmin, vmax], [vmin, vmax], color='#888888',
            linestyle='--', linewidth=1.5, zorder=1)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    # stats (computed but not displayed on plot per user request)

    ax.set_xlabel(r'DFT lattice constant, $a_{\mathrm{DFT}}$ (Å)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Predicted lattice constant, $a_{\mathrm{pred}}$ (Å)',
                  fontsize=14, fontweight='bold')

    # legend
    handles = []
    for mk, disp, short in MODELS:
        if mk in data:
            handles.append(Line2D([0],[0], marker=MARKERS[mk], color='w',
                           markerfacecolor=COLORS[mk], markeredgecolor=COLORS[mk],
                           markersize=9, markeredgewidth=1.5, label=disp))
    handles.append(Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#666', markeredgecolor='#666',
                   markersize=8, label='Si (filled)'))
    handles.append(Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='#666',
                   markersize=8, markeredgewidth=1.8, label='Ge (open)'))
    legend_pub(ax, handles=handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    save_fig(fig, "fig1_lattice_parity")
    plt.close()


# ============================================================================
# FIGURE 2 — Cohesive Energy Parity
# ============================================================================

def fig2_energy_parity(data):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')
    style_ax(ax)

    xs_all, ys_all = [], []

    for mk, disp, short in MODELS:
        if mk not in data:
            continue
        for el in ["Si", "Ge"]:
            d = data[mk].get(el, {})
            if "e_rlx" not in d:
                continue
            # skip DPA-3 (different energy reference ~-107 eV)
            if abs(d["e_rlx"]) > 20:
                continue
            x = GT[el]["dft_e"]
            y = d["e_rlx"]
            xs_all.append(x); ys_all.append(y)
            fc = COLORS[mk] if el == "Si" else "white"
            ax.scatter(x, y, marker=MARKERS[mk], s=140,
                       facecolors=fc, edgecolors=COLORS[mk],
                       linewidth=1.8, zorder=5)

    vmin = min(xs_all + ys_all) - 0.25
    vmax = max(xs_all + ys_all) + 0.25
    ax.plot([vmin, vmax], [vmin, vmax], color='#888888',
            linestyle='--', linewidth=1.5, zorder=1)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    # stats (computed but not displayed on plot per user request)

    ax.set_xlabel(r'DFT cohesive energy, $E_{\mathrm{DFT}}$ (eV/atom)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Predicted energy, $E_{\mathrm{pred}}$ (eV/atom)',
                  fontsize=14, fontweight='bold')

    handles = []
    for mk, disp, short in MODELS:
        if mk in data and mk != "DPA-3-pretrained":
            handles.append(Line2D([0],[0], marker=MARKERS[mk], color='w',
                           markerfacecolor=COLORS[mk], markeredgecolor=COLORS[mk],
                           markersize=9, markeredgewidth=1.5, label=disp))
    handles.append(Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#666', markeredgecolor='#666',
                   markersize=8, label='Si (filled)'))
    handles.append(Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='#666',
                   markersize=8, markeredgewidth=1.8, label='Ge (open)'))
    legend_pub(ax, handles=handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    save_fig(fig, "fig2_energy_parity")
    plt.close()


# ============================================================================
# FIGURE 3 — EOS Curves  (1×2)
# ============================================================================

def fig3_eos_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('white')

    for idx, el in enumerate(["Si", "Ge"]):
        ax = axes[idx]
        style_ax(ax, grid=True)

        for mk, disp, short in MODELS:
            if mk not in data or el not in data[mk]:
                continue
            d = data[mk][el]
            if "eos_a" not in d or "eos_e" not in d:
                continue
            a_arr = np.array(d["eos_a"])
            e_arr = np.array(d["eos_e"])
            # normalize to minimum = 0 so DPA-3 is comparable
            e_arr = e_arr - np.min(e_arr)
            ax.plot(a_arr, e_arr, '-', color=COLORS[mk], linewidth=2.2,
                    label=short, zorder=3)

        # reference lines
        ax.axvline(x=GT[el]["exp_a"], color='#2ca02c', linestyle='--',
                   linewidth=1.8, alpha=0.8, zorder=2)
        ax.axvline(x=GT[el]["dft_a"], color='#9467bd', linestyle=':',
                   linewidth=1.8, alpha=0.8, zorder=2)

        ax.set_xlabel(r'Lattice constant, $a$ (Å)', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r'$E - E_{\min}$ (eV/atom)', fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('')

        # panel label
        panel = '(a)' if idx == 0 else '(b)'
        ax.text(0.08, 0.95, f'{panel} {el}', transform=ax.transAxes,
                fontsize=13, fontweight='bold', verticalalignment='top')

    # shared legend (bottom)
    handles = []
    for mk, disp, short in MODELS:
        if mk in data:
            handles.append(Line2D([0],[0], color=COLORS[mk], linewidth=2.2, label=disp))
    handles.append(Line2D([0],[0], color='#2ca02c', linestyle='--', linewidth=1.8,
                   label=r'$a_{\mathrm{exp}}$'))
    handles.append(Line2D([0],[0], color='#9467bd', linestyle=':', linewidth=1.8,
                   label=r'$a_{\mathrm{DFT}}$'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               framealpha=0.95, edgecolor='black', fancybox=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "fig3_eos_curves")
    plt.close()


# ============================================================================
# FIGURE 4 — Energy–Strain Curves (1×2)
# ============================================================================

def fig4_strain_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('white')

    for idx, el in enumerate(["Si", "Ge"]):
        ax = axes[idx]
        style_ax(ax, grid=True)

        for mk, disp, short in MODELS:
            if mk not in data or el not in data[mk]:
                continue
            d = data[mk][el]
            if "strain_a" not in d or "strain_e" not in d:
                continue
            a_arr = np.array(d["strain_a"])
            e_arr = np.array(d["strain_e"])
            e_arr = e_arr - np.min(e_arr)
            ax.plot(a_arr, e_arr, '-', color=COLORS[mk], linewidth=2.0,
                    label=short, zorder=3)

        ax.axvline(x=GT[el]["exp_a"], color='#2ca02c', linestyle='--',
                   linewidth=1.8, alpha=0.8, zorder=2)
        ax.axvline(x=GT[el]["dft_a"], color='#9467bd', linestyle=':',
                   linewidth=1.8, alpha=0.8, zorder=2)

        ax.set_xlabel(r'Lattice constant, $a$ (Å)', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r'$E - E_{\min}$ (eV/atom)', fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('')

        panel = '(a)' if idx == 0 else '(b)'
        ax.text(0.08, 0.95, f'{panel} {el}', transform=ax.transAxes,
                fontsize=13, fontweight='bold', verticalalignment='top')

    handles = []
    for mk, disp, short in MODELS:
        if mk in data:
            handles.append(Line2D([0],[0], color=COLORS[mk], linewidth=2.0, label=disp))
    handles.append(Line2D([0],[0], color='#2ca02c', linestyle='--', linewidth=1.8,
                   label=r'$a_{\mathrm{exp}}$'))
    handles.append(Line2D([0],[0], color='#9467bd', linestyle=':', linewidth=1.8,
                   label=r'$a_{\mathrm{DFT}}$'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               framealpha=0.95, edgecolor='black', fancybox=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "fig4_strain_curves")
    plt.close()


# ============================================================================
# FIGURE 5 — Lattice Constant Comparison Bar Chart
# ============================================================================

def fig5_lattice_comparison(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    style_ax(ax)

    model_list = [(mk, disp, short) for mk, disp, short in MODELS if mk in data]
    n = len(model_list)
    x_pos = np.arange(n)
    bar_w = 0.35

    si_vals = [data[mk].get("Si", {}).get("a_eos", np.nan) for mk, _, _ in model_list]
    ge_vals = [data[mk].get("Ge", {}).get("a_eos", np.nan) for mk, _, _ in model_list]

    bars_si = ax.bar(x_pos - bar_w/2, si_vals, bar_w,
                     color=[COLORS[mk] for mk, _, _ in model_list],
                     edgecolor='black', linewidth=0.8, alpha=0.90, label='Si')
    bars_ge = ax.bar(x_pos + bar_w/2, ge_vals, bar_w,
                     color=[COLORS[mk] for mk, _, _ in model_list],
                     edgecolor='black', linewidth=0.8, alpha=0.50,
                     hatch='///', label='Ge')

    # reference lines
    ax.axhline(y=GT["Si"]["dft_a"], color='#c41e3a', linestyle='--', linewidth=1.5,
               alpha=0.7, label=r'Si $a_{\mathrm{DFT}}$')
    ax.axhline(y=GT["Ge"]["dft_a"], color='#1a5276', linestyle=':', linewidth=1.5,
               alpha=0.7, label=r'Ge $a_{\mathrm{DFT}}$')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([short for _, _, short in model_list],
                       fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Lattice constant, $a_0$ (Å)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')

    # y-axis limits with room
    all_v = [v for v in si_vals + ge_vals if not np.isnan(v)]
    ymin = min(all_v) - 0.12
    ymax = max(all_v) + 0.12
    ax.set_ylim(ymin, ymax)

    # legend outside data area
    leg_handles = [
        Patch(facecolor='#888', edgecolor='black', linewidth=0.8, alpha=0.90, label='Si'),
        Patch(facecolor='#888', edgecolor='black', linewidth=0.8, alpha=0.50,
              hatch='///', label='Ge'),
        Line2D([0],[0], color='#c41e3a', linestyle='--', linewidth=1.5,
               label=f'Si $a_{{\\mathrm{{DFT}}}}$ = {GT["Si"]["dft_a"]:.3f} Å'),
        Line2D([0],[0], color='#1a5276', linestyle=':', linewidth=1.5,
               label=f'Ge $a_{{\\mathrm{{DFT}}}}$ = {GT["Ge"]["dft_a"]:.3f} Å'),
    ]
    legend_pub(ax, handles=leg_handles, loc='upper right', fontsize=9,
               bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    save_fig(fig, "fig5_lattice_comparison")
    plt.close()


# ============================================================================
# FIGURE 6 — Force Stability Bar Chart
# ============================================================================

def fig6_force_stability(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    style_ax(ax)

    model_list = [(mk, disp, short) for mk, disp, short in MODELS if mk in data]
    n = len(model_list)
    x_pos = np.arange(n)
    bar_w = 0.35

    si_vals, si_errs = [], []
    ge_vals, ge_errs = [], []
    for mk, _, _ in model_list:
        ds = data[mk].get("Si", {})
        si_vals.append(ds.get("mean_maxf", 0))
        si_errs.append(ds.get("std_maxf", 0))
        dg = data[mk].get("Ge", {})
        ge_vals.append(dg.get("mean_maxf", 0))
        ge_errs.append(dg.get("std_maxf", 0))

    ax.bar(x_pos - bar_w/2, si_vals, bar_w, yerr=si_errs,
           capsize=4, color=[COLORS[mk] for mk, _, _ in model_list],
           edgecolor='black', linewidth=0.8, alpha=0.90,
           error_kw=dict(lw=1.5, capthick=1.5))
    ax.bar(x_pos + bar_w/2, ge_vals, bar_w, yerr=ge_errs,
           capsize=4, color=[COLORS[mk] for mk, _, _ in model_list],
           edgecolor='black', linewidth=0.8, alpha=0.50, hatch='///',
           error_kw=dict(lw=1.5, capthick=1.5))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([short for _, _, short in model_list],
                       fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Mean max $|F|$ (eV/Å)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')

    # legend placed at top-left above the tallest bars (classical)
    leg_handles = [
        Patch(facecolor='#888', edgecolor='black', alpha=0.90, label='Si'),
        Patch(facecolor='#888', edgecolor='black', alpha=0.50, hatch='///', label='Ge'),
    ]
    legend_pub(ax, handles=leg_handles, loc='upper left', fontsize=10,
               bbox_to_anchor=(0.01, 0.99))

    plt.tight_layout()
    save_fig(fig, "fig6_force_stability")
    plt.close()


# ============================================================================
# FIGURE 7 — Per-Model Lattice Parity (2×3 grid)
# ============================================================================

def fig7_per_model_lattice(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    fig.patch.set_facecolor('white')

    global_min, global_max = 5.25, 5.90

    for idx, (mk, disp, short) in enumerate(MODELS):
        r, c = divmod(idx, 3)
        ax = axes[r][c]
        style_ax(ax)

        ax.plot([global_min, global_max], [global_min, global_max],
                color='#888888', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

        if mk not in data:
            ax.text(0.5, 0.5, f'{disp}\n(no data)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            continue

        color = COLORS[mk]
        marker = MARKERS[mk]
        xs, ys = [], []

        for el in ["Si", "Ge"]:
            d = data[mk].get(el, {})
            gt_val = GT[el]["dft_a"]
            pred_val = d.get("a_eos")
            if pred_val is None:
                continue
            fc = color if el == "Si" else "white"
            ax.scatter(gt_val, pred_val, marker=marker, s=200,
                       facecolors=fc, edgecolors=color, linewidth=2.0, zorder=5)
            ax.annotate(el, (gt_val, pred_val),
                        textcoords="offset points", xytext=(10, -5),
                        fontsize=11, fontweight='bold', color='#333333')
            xs.append(gt_val); ys.append(pred_val)

        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$a_{\mathrm{DFT}}$ (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$a_{\mathrm{pred}}$ (Å)', fontsize=12, fontweight='bold')

        # model name as panel label
        ax.text(0.04, 0.96, disp, transform=ax.transAxes, fontsize=11,
                fontweight='bold', verticalalignment='top', color=color)

        if len(xs) >= 2:
            xa, ya = np.array(xs), np.array(ys)
            ss_res = np.sum((ya - xa)**2)
            ss_tot = np.sum((ya - np.mean(ya))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((ya - xa)**2))
            mae = np.mean(np.abs(ya - xa))
            stats_str = f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f} Å\nMAE = {mae:.4f} Å'
            ax.text(0.04, 0.82, stats_str, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                              edgecolor='none', alpha=0.12))

    plt.tight_layout()
    save_fig(fig, "fig7_per_model_lattice")
    plt.close()


# ============================================================================
# FIGURE 8 — Combined Correlation (1×3)
# ============================================================================

def fig8_combined_correlation(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')

    # ── Panel (a): Lattice Constant Parity ──
    ax = axes[0]
    style_ax(ax)
    xs, ys = [], []
    for mk, disp, short in MODELS:
        if mk not in data: continue
        for el in ["Si", "Ge"]:
            d = data[mk].get(el, {})
            if "a_eos" not in d: continue
            x, y = GT[el]["dft_a"], d["a_eos"]
            xs.append(x); ys.append(y)
            fc = COLORS[mk] if el == "Si" else "white"
            ax.scatter(x, y, marker=MARKERS[mk], s=120,
                       facecolors=fc, edgecolors=COLORS[mk], linewidth=1.6, zorder=5)
    vmin = min(xs + ys) - 0.06; vmax = max(xs + ys) + 0.06
    ax.plot([vmin, vmax], [vmin, vmax], color='#888', linestyle='--', lw=1.5, zorder=1)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax); ax.set_aspect('equal')
    # stats removed per user request
    ax.set_xlabel(r'$a_{\mathrm{DFT}}$ (Å)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$a_{\mathrm{pred}}$ (Å)', fontsize=13, fontweight='bold')
    ax.text(0.97, 0.03, '(a)', transform=ax.transAxes, fontsize=13,
            fontweight='bold', ha='right')

    # ── Panel (b): Cohesive Energy Parity ──
    ax = axes[1]
    style_ax(ax)
    xs, ys = [], []
    for mk, disp, short in MODELS:
        if mk not in data: continue
        for el in ["Si", "Ge"]:
            d = data[mk].get(el, {})
            if "e_rlx" not in d or abs(d["e_rlx"]) > 20: continue
            x, y = GT[el]["dft_e"], d["e_rlx"]
            xs.append(x); ys.append(y)
            fc = COLORS[mk] if el == "Si" else "white"
            ax.scatter(x, y, marker=MARKERS[mk], s=120,
                       facecolors=fc, edgecolors=COLORS[mk], linewidth=1.6, zorder=5)
    vmin = min(xs + ys) - 0.2; vmax = max(xs + ys) + 0.2
    ax.plot([vmin, vmax], [vmin, vmax], color='#888', linestyle='--', lw=1.5, zorder=1)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax); ax.set_aspect('equal')
    # stats removed per user request
    ax.set_xlabel(r'$E_{\mathrm{DFT}}$ (eV/atom)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$E_{\mathrm{pred}}$ (eV/atom)', fontsize=13, fontweight='bold')
    ax.text(0.97, 0.03, '(b)', transform=ax.transAxes, fontsize=13,
            fontweight='bold', ha='right')

    # ── Panel (c): Force Stability Grouped Bar ──
    ax = axes[2]
    style_ax(ax)
    model_list = [(mk, disp, short) for mk, disp, short in MODELS if mk in data]
    nn = len(model_list)
    xp = np.arange(nn)
    bw = 0.35

    si_v = [data[mk].get("Si", {}).get("mean_maxf", 0) for mk, _, _ in model_list]
    si_e = [data[mk].get("Si", {}).get("std_maxf", 0) for mk, _, _ in model_list]
    ge_v = [data[mk].get("Ge", {}).get("mean_maxf", 0) for mk, _, _ in model_list]
    ge_e = [data[mk].get("Ge", {}).get("std_maxf", 0) for mk, _, _ in model_list]

    ax.bar(xp - bw/2, si_v, bw, yerr=si_e, capsize=3,
           color=[COLORS[mk] for mk, _, _ in model_list],
           edgecolor='black', linewidth=0.7, alpha=0.88,
           error_kw=dict(lw=1.2, capthick=1.2))
    ax.bar(xp + bw/2, ge_v, bw, yerr=ge_e, capsize=3,
           color=[COLORS[mk] for mk, _, _ in model_list],
           edgecolor='black', linewidth=0.7, alpha=0.45, hatch='///',
           error_kw=dict(lw=1.2, capthick=1.2))

    ax.set_xticks(xp)
    ax.set_xticklabels([short for _, _, short in model_list],
                       fontsize=10, fontweight='bold', rotation=30, ha='right')
    ax.set_ylabel(r'Mean max $|F|$ (eV/Å)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')

    leg = [Patch(facecolor='#888', edgecolor='black', alpha=0.88, label='Si'),
           Patch(facecolor='#888', edgecolor='black', alpha=0.45, hatch='///', label='Ge')]
    legend_pub(ax, handles=leg, loc='upper right', fontsize=9,
               bbox_to_anchor=(0.99, 0.99))
    ax.text(0.03, 0.97, '(c)', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top')

    # shared model legend across bottom
    m_handles = []
    for mk, disp, short in MODELS:
        if mk in data:
            m_handles.append(Line2D([0],[0], marker=MARKERS[mk], color='w',
                             markerfacecolor=COLORS[mk], markeredgecolor=COLORS[mk],
                             markersize=8, markeredgewidth=1.4, label=disp))
    m_handles.append(Line2D([0],[0], marker='o', color='w',
                     markerfacecolor='#666', markersize=7, label='Si (filled)'))
    m_handles.append(Line2D([0],[0], marker='o', color='w',
                     markerfacecolor='white', markeredgecolor='#666',
                     markersize=7, markeredgewidth=1.6, label='Ge (open)'))
    fig.legend(handles=m_handles, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.95, edgecolor='black', fancybox=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, "fig8_combined_correlation")
    plt.close()


# ============================================================================
# FIGURE 9 — Optuna Convergence
# ============================================================================

def fig9_optuna_convergence(optuna):
    if optuna is None:
        print("  ⚠  Skipping fig9 (no Optuna data)")
        return

    trials = optuna.get("all_trials", [])
    if not trials:
        print("  ⚠  Skipping fig9 (empty trials)")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('white')
    style_ax(ax, grid=True)

    # colour cycle
    cmap = plt.cm.get_cmap('tab10')
    for t in trials:
        tid = t["trial"]
        losses = t.get("train_losses", [])
        if not losses:
            continue
        steps = np.arange(1, len(losses) + 1) * 100  # disp_freq = 100
        ax.plot(steps, losses, '-o', color=cmap(tid % 10), linewidth=1.8,
                markersize=4, markeredgewidth=0.6, markeredgecolor='black',
                alpha=0.85, label=f'Trial {tid}')

    ax.set_xlabel(r'Training step', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'RMSE($F$) (eV/Å)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')

    best = optuna.get("best_trial", "?")
    best_val = optuna.get("best_value", "?")
    ax.text(0.97, 0.97,
            f'Best: Trial {best}\nRMSE($F$) = {best_val}',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                      edgecolor='#ccc', alpha=0.92))

    legend_pub(ax, loc='upper left', fontsize=9, ncol=2,
               bbox_to_anchor=(0.12, 0.99))

    plt.tight_layout()
    save_fig(fig, "fig9_optuna_convergence")
    plt.close()


# ============================================================================
# SAVE DATA TABLES (CSV)
# ============================================================================

def save_summary_csv(data):
    """Save a summary CSV alongside the figures."""
    import csv
    rows = []
    for mk, disp, short in MODELS:
        if mk not in data: continue
        for el in ["Si", "Ge"]:
            d = data[mk].get(el, {})
            if not d: continue
            gt = GT[el]
            a_eos = d.get("a_eos", np.nan)
            err_dft = abs(a_eos - gt["dft_a"]) / gt["dft_a"] * 100
            rows.append({
                "Model": disp,
                "Type": "MLIP" if mk not in ["Tersoff", "SW"] else "Classical",
                "Element": el,
                "a_EOS_A": f"{a_eos:.4f}",
                "a_DFT_A": f"{gt['dft_a']:.3f}",
                "Err_DFT_pct": f"{err_dft:.2f}",
                "E_rlx_eV": f"{d.get('e_rlx', np.nan):.4f}",
                "MeanMaxF_eV_A": f"{d.get('mean_maxf', np.nan):.4f}",
                "StdMaxF_eV_A": f"{d.get('std_maxf', np.nan):.4f}",
            })
    outf = os.path.join(OUT_DIR, "summary_table.csv")
    with open(outf, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ summary_table.csv")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 72)
    print("  PUBLICATION-QUALITY PLOTS — Si & Ge Benchmark")
    print("=" * 72)

    os.makedirs(OUT_DIR, exist_ok=True)

    data, optuna = load_data()
    print(f"  Models loaded: {list(data.keys())}")

    print("\nGenerating figures …")
    fig1_lattice_parity(data)
    fig2_energy_parity(data)
    fig3_eos_curves(data)
    fig4_strain_curves(data)
    fig5_lattice_comparison(data)
    fig6_force_stability(data)
    fig7_per_model_lattice(data)
    fig8_combined_correlation(data)
    fig9_optuna_convergence(optuna)
    save_summary_csv(data)

    print("\n" + "=" * 72)
    print("  ✅  ALL PUBLICATION PLOTS GENERATED")
    print(f"  Output: {OUT_DIR}")
    print("=" * 72 + "\n")
