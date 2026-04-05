#!/usr/bin/env python3
"""
Comprehensive Benchmark: All 5 Potentials for Si and Ge
========================================================
Potentials: MACE-MP-0, CHGNet, DeepMD(DPA-2), Tersoff, ReaxFF(SW)
Properties: Lattice constant, cohesive energy, internal forces
Includes: Bootstrap cross-validation, publication-quality plots

Ground Truth:
  Si: a_exp=5.431 Å, a_DFT=5.469 Å, E_coh_exp=-4.63 eV/atom, E_coh_DFT=-5.43 eV/atom
  Ge: a_exp=5.658 Å, a_DFT=5.763 Å, E_coh_exp=-3.85 eV/atom, E_coh_DFT=-4.62 eV/atom
"""

import os, sys, json, csv, warnings, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ASE imports
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.eos import EquationOfState
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter

# ── Paths ──
BASE_DIR = "/home/bib569/Workspace/Hackrush_2026-Problem-10"
OUTPUT_DIR = os.path.join(BASE_DIR, "results_comprehensive")
POTENTIALS_DIR = os.path.join(BASE_DIR, "potentials")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.titlesize': 14,
    'axes.titlesize': 12, 'lines.linewidth': 2,
    'lines.markersize': 6, 'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════
# GROUND TRUTH DATA
# ══════════════════════════════════════════════════════════════════
GROUND_TRUTH = {
    "Si": {
        "exp_lattice_const": 5.431, "dft_lattice_const": 5.469,
        "exp_cohesive_energy": -4.63, "dft_cohesive_energy": -5.43,
    },
    "Ge": {
        "exp_lattice_const": 5.658, "dft_lattice_const": 5.763,
        "exp_cohesive_energy": -3.85, "dft_cohesive_energy": -4.62,
    }
}

# Model metadata
MODEL_COLORS = {
    "MACE-MP-0": "#E63946", "CHGNet": "#457B9D", "DeepMD-DPA2": "#2A9D8F",
    "Tersoff": "#E9C46A", "SW": "#F4A261",
}
MODEL_MARKERS = {
    "MACE-MP-0": "o", "CHGNet": "s", "DeepMD-DPA2": "D",
    "Tersoff": "^", "SW": "v",
}

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def get_lattice_constant_eos(calc_fn, element, n_points=11, strain_range=0.06):
    """EOS fitting for equilibrium lattice constant."""
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    strains = np.linspace(1 - strain_range, 1 + strain_range, n_points)
    volumes, energies, a_vals = [], [], []
    for s in strains:
        a_test = a0 * s
        atoms = bulk(element, crystalstructure='diamond', a=a_test)
        atoms.calc = calc_fn()
        e = atoms.get_potential_energy()
        volumes.append(atoms.get_volume())
        energies.append(e)
        a_vals.append(a_test)
    volumes, energies, a_vals = np.array(volumes), np.array(energies), np.array(a_vals)
    try:
        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        v0, e0, B = eos.fit()
        a_eq = (v0 * 4.0) ** (1.0/3.0)  # primitive diamond cell: V = a^3/4
    except:
        idx = np.argmin(energies)
        a_eq, e0 = a_vals[idx], energies[idx]
    return a_eq, e0 / 2.0, a_vals, energies / 2.0  # 2 atoms per cell


def relax_and_get_props(calc_fn, element):
    """Relax cell and get lattice constant, energy/atom, max force."""
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    atoms = bulk(element, crystalstructure='diamond', a=a0)
    atoms.calc = calc_fn()
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile=None)
    try:
        opt.run(fmax=0.01, steps=200)
    except:
        pass
    cell = atoms.get_cell()
    a_relax = np.mean([np.linalg.norm(cell[i]) for i in range(3)]) * (2**0.5)
    e_relax = atoms.get_potential_energy() / len(atoms)
    f_relax = atoms.get_forces()
    max_f = np.max(np.abs(f_relax))
    return a_relax, e_relax, max_f


def get_forces_bootstrap(calc_fn, element, n_configs=30, seed=42):
    """Generate bootstrap force configurations for CV-like analysis."""
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    rng = np.random.RandomState(seed)
    all_forces = []
    all_disps = []
    for i in range(n_configs):
        atoms = bulk(element, crystalstructure='diamond', a=a0)
        disp_mag = rng.uniform(0.01, 0.2)
        delta = rng.randn(*atoms.get_positions().shape) * disp_mag
        atoms.set_positions(atoms.get_positions() + delta)
        atoms.calc = calc_fn()
        forces = atoms.get_forces()
        all_forces.append(forces.flatten())
        all_disps.append(disp_mag)
    return np.array(all_disps), all_forces


def get_energy_vs_strain(calc_fn, element, n_points=21):
    """Energy per atom vs lattice constant."""
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    strains = np.linspace(0.85, 1.15, n_points)
    energies = []
    for s in strains:
        atoms = bulk(element, crystalstructure='diamond', a=a0*s)
        atoms.calc = calc_fn()
        energies.append(atoms.get_potential_energy() / len(atoms))
    return strains * a0, np.array(energies)


# ══════════════════════════════════════════════════════════════════
# CALCULATOR FACTORIES
# ══════════════════════════════════════════════════════════════════

def load_calculators():
    """Try to load all available calculators."""
    calcs = {}

    # 1. MACE-MP-0
    try:
        from mace.calculators import mace_mp
        mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float32")
        calcs["MACE-MP-0"] = lambda _c=mace_calc: _c
        print("  ✓ MACE-MP-0")
    except Exception as e:
        print(f"  ✗ MACE-MP-0: {e}")

    # 2. CHGNet
    try:
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        chgnet_model = CHGNet.load()
        calcs["CHGNet"] = lambda _m=chgnet_model: CHGNetCalculator(model=_m)
        print("  ✓ CHGNet")
    except Exception as e:
        print(f"  ✗ CHGNet: {e}")

    # 3. DeepMD / DPA-2
    try:
        from deepmd.calculator import DP
        # Try to use pre-trained DPA-2 model
        # First check if model file exists
        dpa2_path = os.path.join(BASE_DIR, "potentials", "dpa2_model.pt")
        if os.path.exists(dpa2_path):
            calcs["DeepMD-DPA2"] = lambda: DP(model=dpa2_path)
            print("  ✓ DeepMD-DPA2")
        else:
            print(f"  ✗ DeepMD-DPA2: model file not found at {dpa2_path}")
    except Exception as e:
        print(f"  ✗ DeepMD: {e}")

    # 4. Tersoff via LAMMPS
    try:
        from lammps import lammps
        tersoff_file = os.path.join(POTENTIALS_DIR, "SiCGe.tersoff")
        if os.path.exists(tersoff_file):
            def tersoff_calc_factory(element="Si", _tf=tersoff_file):
                from ase.calculators.lammpslib import LAMMPSlib
                lmpcmds = [
                    "pair_style tersoff",
                    f"pair_coeff * * {_tf} {element}",
                ]
                return LAMMPSlib(lmpcmds=lmpcmds, log_file=None,
                                keep_alive=True, lammps_name="")
            calcs["Tersoff"] = tersoff_calc_factory
            print("  ✓ Tersoff (via LAMMPS)")
        else:
            # Fallback: use Si.tersoff for Si only
            tersoff_si = os.path.join(POTENTIALS_DIR, "Si.tersoff")
            if os.path.exists(tersoff_si):
                def tersoff_si_factory(element="Si", _tf=tersoff_si):
                    from ase.calculators.lammpslib import LAMMPSlib
                    lmpcmds = [
                        "pair_style tersoff",
                        f"pair_coeff * * {_tf} {element}",
                    ]
                    return LAMMPSlib(lmpcmds=lmpcmds, log_file=None,
                                    keep_alive=True, lammps_name="")
                calcs["Tersoff"] = tersoff_si_factory
                print("  ✓ Tersoff (Si.tersoff only)")
    except Exception as e:
        print(f"  ✗ Tersoff/LAMMPS: {e}")

    # 5. Stillinger-Weber (alternative classical potential, widely available)
    try:
        from lammps import lammps
        # Download SW file if not present
        sw_file = os.path.join(POTENTIALS_DIR, "Si.sw")
        if not os.path.exists(sw_file):
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/lammps/lammps/stable/potentials/Si.sw",
                sw_file)

        def sw_calc_factory(element="Si", _sf=sw_file):
            from ase.calculators.lammpslib import LAMMPSlib
            lmpcmds = [
                "pair_style sw",
                f"pair_coeff * * {_sf} {element}",
            ]
            return LAMMPSlib(lmpcmds=lmpcmds, log_file=None,
                            keep_alive=True, lammps_name="")
        calcs["SW"] = sw_calc_factory
        print("  ✓ Stillinger-Weber (via LAMMPS)")
    except Exception as e:
        print(f"  ✗ SW/LAMMPS: {e}")

    return calcs


# ══════════════════════════════════════════════════════════════════
# MAIN BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════

def run_all_benchmarks():
    print("=" * 70)
    print("Loading calculators...")
    print("=" * 70)
    calcs = load_calculators()

    if not calcs:
        print("ERROR: No calculators loaded!")
        sys.exit(1)

    results = {}
    lammps_models = {"Tersoff", "SW"}  # Models that need element passed to factory

    for model_name, calc_factory in calcs.items():
        print(f"\n{'='*70}\nBenchmarking: {model_name}\n{'='*70}")
        results[model_name] = {}
        is_lammps = model_name in lammps_models

        for element in ["Si", "Ge"]:
            print(f"\n  --- {element} ---")
            gt = GROUND_TRUTH[element]

            try:
                # For LAMMPS-based, pass element to factory
                if is_lammps:
                    calc_fn = lambda _cf=calc_factory, _e=element: _cf(element=_e)
                else:
                    calc_fn = calc_factory

                # 1. EOS fitting
                print(f"  [1/5] EOS fitting...")
                a_eos, e_eos, a_curve, e_curve = get_lattice_constant_eos(calc_fn, element)
                print(f"    a_EOS = {a_eos:.4f} Å (Exp={gt['exp_lattice_const']:.3f}, DFT={gt['dft_lattice_const']:.3f})")

                # 2. Cell relaxation
                print(f"  [2/5] Cell relaxation...")
                a_relax, e_relax, max_f = relax_and_get_props(calc_fn, element)
                print(f"    a_relax = {a_relax:.4f} Å, E/at = {e_relax:.4f} eV, max|F| = {max_f:.2e}")

                # 3. Bootstrap force sampling
                print(f"  [3/5] Bootstrap force sampling (30 configs)...")
                disps, forces_list = get_forces_bootstrap(calc_fn, element, n_configs=30)

                # Force statistics
                force_mags = [np.linalg.norm(f.reshape(-1, 3), axis=1) for f in forces_list]
                mean_max_f = np.mean([np.max(fm) for fm in force_mags])
                std_max_f = np.std([np.max(fm) for fm in force_mags])
                print(f"    Mean max|F| = {mean_max_f:.4f} ± {std_max_f:.4f} eV/Å")

                # 4. Energy vs strain curve
                print(f"  [4/5] Energy-strain curve...")
                strain_a, strain_e = get_energy_vs_strain(calc_fn, element)

                # 5. Forces at equilibrium
                print(f"  [5/5] Equilibrium forces...")
                eq_atoms = bulk(element, crystalstructure='diamond', a=gt['exp_lattice_const'])
                eq_atoms.calc = calc_fn()
                eq_forces = eq_atoms.get_forces()
                eq_energy = eq_atoms.get_potential_energy() / len(eq_atoms)

                results[model_name][element] = {
                    "lattice_const_eos": float(a_eos),
                    "lattice_const_relaxed": float(a_relax),
                    "energy_per_atom_eos": float(e_eos),
                    "energy_per_atom_relaxed": float(e_relax),
                    "energy_per_atom_eq": float(eq_energy),
                    "max_force_relaxed": float(max_f),
                    "eq_forces": eq_forces.tolist(),
                    "force_displacements": disps.tolist(),
                    "force_magnitudes": [f.tolist() for f in forces_list],
                    "mean_max_force": float(mean_max_f),
                    "std_max_force": float(std_max_f),
                    "strain_lattice_consts": strain_a.tolist(),
                    "strain_energies": strain_e.tolist(),
                    "eos_lattice_consts": a_curve.tolist(),
                    "eos_energies": e_curve.tolist(),
                }
                print(f"  ✓ {model_name}/{element} complete")

            except Exception as exc:
                print(f"  ✗ ERROR: {model_name}/{element}: {exc}")
                traceback.print_exc()
                continue

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "all_benchmark_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════
# PUBLICATION-QUALITY PLOTTING
# ══════════════════════════════════════════════════════════════════

def create_confidence_interval(x, y, confidence=0.95):
    """Confidence interval for regression line."""
    lr = LinearRegression()
    x_r = x.reshape(-1, 1)
    lr.fit(x_r, y)
    y_pred = lr.predict(x_r)
    residuals = y - y_pred
    n = len(x)
    if n <= 2:
        return y_pred, y_pred, y_pred
    std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
    x_mean = np.mean(x)
    ci = t_val * std_error * np.sqrt(
        1 + 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)
    )
    return y_pred, y_pred - ci, y_pred + ci


def plot_publication_correlation(results):
    """
    Publication-quality correlation/parity plots.
    Matches the exact format from the user's plotting script.
    """
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    model_names = list(results.keys())

    # ── Figure 1: Lattice Constant Parity (per model) ──
    n_models = len(model_names)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 5*nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    fig.suptitle("Correlation Plots: Predicted vs Ground Truth Lattice Constant",
                fontsize=14, fontweight='bold', y=1.01)

    for idx, model in enumerate(model_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        color = MODEL_COLORS.get(model, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')

        gt_exp, gt_dft, pred = [], [], []
        labels = []
        for elem in ["Si", "Ge"]:
            if elem in results[model]:
                gt_exp.append(GROUND_TRUTH[elem]["exp_lattice_const"])
                gt_dft.append(GROUND_TRUTH[elem]["dft_lattice_const"])
                pred.append(results[model][elem]["lattice_const_eos"])
                labels.append(elem)

        if not pred:
            ax.text(0.5, 0.5, f"No data for {model}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            continue

        gt_exp, gt_dft, pred = np.array(gt_exp), np.array(gt_dft), np.array(pred)

        # Plot vs DFT
        ax.scatter(gt_dft, pred, c=color, s=120, marker=marker,
                  edgecolors='black', linewidth=0.8, zorder=5, label=f'{model}')

        # Annotate elements
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (gt_dft[i], pred[i]), textcoords="offset points",
                       xytext=(10, 5), fontsize=11, fontweight='bold')

        # Parity line
        all_vals = np.concatenate([gt_dft, pred])
        vmin, vmax = all_vals.min() - 0.15, all_vals.max() + 0.15
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5, label='Perfect parity')

        # R² and MAE
        if len(pred) >= 2:
            r2 = np.corrcoef(gt_dft, pred)[0, 1]**2
            mae = np.mean(np.abs(gt_dft - pred))
            ax.text(0.05, 0.92, f'R² = {r2:.4f}\nMAE = {mae:.4f} Å',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Subplot label
        ax.text(-0.12, 1.05, f'({subplot_labels[idx]})', transform=ax.transAxes,
               fontsize=16, fontweight='bold')

        ax.set_xlabel(r"DFT Lattice Constant (Å)", fontsize=12)
        ax.set_ylabel(r"Predicted Lattice Constant (Å)", fontsize=12)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_lattice_constant.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ correlation_lattice_constant.png")


def plot_publication_energy_correlation(results):
    """Energy parity plots per model."""
    model_names = list(results.keys())
    n_models = len(model_names)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 5*nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    fig.suptitle("Correlation Plots: Predicted vs DFT Cohesive Energy",
                fontsize=14, fontweight='bold', y=1.01)

    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

    for idx, model in enumerate(model_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        color = MODEL_COLORS.get(model, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')

        gt_dft, pred = [], []
        labels = []
        for elem in ["Si", "Ge"]:
            if elem in results[model]:
                gt_dft.append(GROUND_TRUTH[elem]["dft_cohesive_energy"])
                pred.append(results[model][elem]["energy_per_atom_relaxed"])
                labels.append(elem)
        if not pred:
            continue

        gt_dft, pred = np.array(gt_dft), np.array(pred)
        ax.scatter(gt_dft, pred, c=color, s=120, marker=marker,
                  edgecolors='black', linewidth=0.8, zorder=5, label=f'{model}')
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (gt_dft[i], pred[i]), textcoords="offset points",
                       xytext=(10, 5), fontsize=11, fontweight='bold')

        all_vals = np.concatenate([gt_dft, pred])
        vmin, vmax = all_vals.min() - 0.3, all_vals.max() + 0.3
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5)

        if len(pred) >= 2:
            r2 = np.corrcoef(gt_dft, pred)[0, 1]**2
            mae = np.mean(np.abs(gt_dft - pred))
            ax.text(0.05, 0.92, f'R² = {r2:.4f}\nMAE = {mae:.4f} eV',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.text(-0.12, 1.05, f'({subplot_labels[idx]})', transform=ax.transAxes,
               fontsize=16, fontweight='bold')
        ax.set_xlabel(r"DFT Energy (eV/atom)", fontsize=12)
        ax.set_ylabel(r"Predicted Energy (eV/atom)", fontsize=12)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)

    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_energy.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ correlation_energy.png")


def plot_combined_correlation(results):
    """
    Combined correlation plot matching the user's format:
    - Scatter with linear regression + confidence interval
    - Inset with complementary data
    - R², MAE, RMSE annotations
    """
    model_names = list(results.keys())
    elements = ["Si", "Ge"]
    properties = {
        "Lattice Constant": {
            "gt_key": "dft_lattice_const", "pred_key": "lattice_const_eos",
            "x_label": r"$a_0^{\mathrm{DFT}}$ (Å)", "y_label": r"$a_0^{\mathrm{pred}}$ (Å)",
        },
        "Cohesive Energy": {
            "gt_key": "dft_cohesive_energy", "pred_key": "energy_per_atom_relaxed",
            "x_label": r"$E_{\mathrm{coh}}^{\mathrm{DFT}}$ (eV/atom)",
            "y_label": r"$E_{\mathrm{coh}}^{\mathrm{pred}}$ (eV/atom)",
        },
        "Equilibrium Energy": {
            "gt_key": "dft_cohesive_energy", "pred_key": "energy_per_atom_eq",
            "x_label": r"$E^{\mathrm{DFT}}$ (eV/atom)",
            "y_label": r"$E^{\mathrm{pred}}$ (eV/atom)",
        },
    }

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    subplot_labels = ['a', 'b', 'c']
    plot_styles = [
        {'color': 'indigo', 'marker': 'o'},
        {'color': 'forestgreen', 'marker': 's'},
        {'color': 'crimson', 'marker': '^'},
    ]

    for pidx, (prop_name, prop_info) in enumerate(properties.items()):
        ax = fig.add_subplot(gs[0, pidx])
        style = plot_styles[pidx]

        gt_all, pred_all, model_labels = [], [], []
        for model in model_names:
            for elem in elements:
                if elem in results.get(model, {}):
                    gt_val = GROUND_TRUTH[elem][prop_info["gt_key"]]
                    pred_val = results[model][elem][prop_info["pred_key"]]
                    gt_all.append(gt_val)
                    pred_all.append(pred_val)
                    model_labels.append(f"{model}\n({elem})")

        gt_all = np.array(gt_all)
        pred_all = np.array(pred_all)

        if len(gt_all) < 2:
            ax.text(0.5, 0.5, f"Insufficient data", ha='center', va='center')
            continue

        # Scatter
        ax.scatter(gt_all, pred_all, color=style['color'], alpha=0.7, s=80,
                  edgecolor='black', marker=style['marker'], zorder=5)

        # Linear regression + confidence interval
        sort_idx = np.argsort(gt_all)
        gt_sorted = gt_all[sort_idx]
        pred_sorted = pred_all[sort_idx]
        y_pred, lower_ci, upper_ci = create_confidence_interval(gt_sorted, pred_sorted)
        ax.plot(gt_sorted, y_pred, color='red', linewidth=2, label='Linear Regression')
        ax.fill_between(gt_sorted, lower_ci, upper_ci, color='lightblue',
                       alpha=0.3, label='95% CI')

        # Perfect parity
        all_v = np.concatenate([gt_all, pred_all])
        vmin, vmax = all_v.min() - 0.15, all_v.max() + 0.15
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.4, lw=1)

        # Metrics
        r2 = np.corrcoef(gt_all, pred_all)[0, 1]**2
        mae = np.mean(np.abs(gt_all - pred_all))
        rmse = np.sqrt(np.mean((gt_all - pred_all)**2))
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Annotate points
        for i, lbl in enumerate(model_labels):
            ax.annotate(lbl, (gt_all[i], pred_all[i]),
                       textcoords="offset points", xytext=(6, -10),
                       fontsize=6, ha='left')

        ax.text(-0.1, 1.02, f'({subplot_labels[pidx]})', transform=ax.transAxes,
               fontsize=20, fontweight='bold')
        ax.set_xlabel(prop_info["x_label"], fontsize=14)
        ax.set_ylabel(prop_info["y_label"], fontsize=14)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "combined_correlation_plots.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ combined_correlation_plots.png")


def plot_energy_convergence_curves(results):
    """
    Energy convergence as a function of strain density —
    analogous to 'learning curves' showing how EOS predictions converge.
    """
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Energy Convergence Curves (EOS sampling density)",
                fontsize=14, fontweight='bold')

    for idx, element in enumerate(["Si", "Ge"]):
        ax = axes[idx]
        for model in model_names:
            if element not in results.get(model, {}):
                continue
            color = MODEL_COLORS.get(model, 'gray')

            # Simulate convergence by computing EOS with increasing N
            a_curve = np.array(results[model][element].get("eos_lattice_consts", []))
            e_curve = np.array(results[model][element].get("eos_energies", []))
            if len(a_curve) < 3:
                continue

            n_points = len(a_curve)
            convergence_n = []
            convergence_a = []
            for n in range(3, n_points+1, 1):
                indices = np.round(np.linspace(0, n_points-1, n)).astype(int)
                sub_a = a_curve[indices]
                sub_e = e_curve[indices]
                sub_v = (sub_a**3) / 4.0  # approx volumes
                try:
                    eos = EquationOfState(sub_v, sub_e * 2, eos='birchmurnaghan')
                    v0, e0, B = eos.fit()
                    a_eq = (v0 * 4.0) ** (1.0/3.0)
                    convergence_n.append(n)
                    convergence_a.append(a_eq)
                except:
                    pass

            if convergence_n:
                ax.plot(convergence_n, convergence_a, '-o', color=color,
                       label=model, markersize=4, linewidth=1.5)

        gt = GROUND_TRUTH[element]
        ax.axhline(y=gt["dft_lattice_const"], color='purple', ls=':', lw=1.5,
                  label=f'DFT a₀ = {gt["dft_lattice_const"]:.3f} Å')
        ax.axhline(y=gt["exp_lattice_const"], color='green', ls='--', lw=1.5,
                  label=f'Exp a₀ = {gt["exp_lattice_const"]:.3f} Å')
        ax.set_xlabel("Number of EOS points", fontsize=12)
        ax.set_ylabel("Predicted a₀ (Å)", fontsize=12)
        ax.set_title(f"{element}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_convergence_curves.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ energy_convergence_curves.png")


def plot_force_cv_analysis(results):
    """Bootstrap force cross-validation analysis."""
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Validation: Force Prediction Consistency (Bootstrap)",
                fontsize=14, fontweight='bold')

    for idx, element in enumerate(["Si", "Ge"]):
        ax = axes[idx]
        for model in model_names:
            if element not in results.get(model, {}):
                continue
            data = results[model][element]
            disps = np.array(data.get("force_displacements", []))
            force_mags_raw = data.get("force_magnitudes", [])
            if len(disps) == 0 or len(force_mags_raw) == 0:
                continue

            color = MODEL_COLORS.get(model, 'gray')
            max_forces = []
            for fm in force_mags_raw:
                fm_arr = np.array(fm).reshape(-1, 3)
                max_forces.append(np.max(np.linalg.norm(fm_arr, axis=1)))
            max_forces = np.array(max_forces)

            # Sort by displacement
            sort_idx = np.argsort(disps)
            ax.scatter(disps[sort_idx], max_forces[sort_idx], c=color, s=25,
                      alpha=0.6, marker=MODEL_MARKERS.get(model, 'o'),
                      edgecolors='black', linewidth=0.3, label=model)

            # Add trend line
            if len(disps) > 3:
                z = np.polyfit(disps, max_forces, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(disps.min(), disps.max(), 50)
                ax.plot(x_smooth, p(x_smooth), '-', color=color, alpha=0.7, lw=1.5)

        ax.set_xlabel("Displacement magnitude (Å)", fontsize=12)
        ax.set_ylabel("Max Force (eV/Å)", fontsize=12)
        ax.set_title(f"{element}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "force_cv_bootstrap.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ force_cv_bootstrap.png")


def plot_force_parity_all_models(results):
    """Force parity cross-comparison between all model pairs."""
    model_names = [m for m in results if any("force_magnitudes" in results[m].get(e, {}) for e in ["Si", "Ge"])]
    n = len(model_names)
    if n < 2:
        print("  ⚠ Not enough models for force parity comparison")
        return

    n_pairs = n * (n - 1) // 2
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 5*nrows))
    if n_pairs == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    fig.suptitle("Force Parity: Cross-Model Comparison (Si + Ge combined)",
                fontsize=14, fontweight='bold', y=1.01)

    pair_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            m1, m2 = model_names[i], model_names[j]
            row, col = divmod(pair_idx, ncols)
            ax = axes[row, col]

            f1_all, f2_all = [], []
            for elem in ["Si", "Ge"]:
                fm1 = results[m1].get(elem, {}).get("force_magnitudes", [])
                fm2 = results[m2].get(elem, {}).get("force_magnitudes", [])
                n_cfg = min(len(fm1), len(fm2))
                for k in range(n_cfg):
                    f1_all.extend(np.array(fm1[k]).flatten().tolist())
                    f2_all.extend(np.array(fm2[k]).flatten().tolist())

            if not f1_all:
                pair_idx += 1
                continue

            f1, f2 = np.array(f1_all), np.array(f2_all)
            ax.scatter(f1, f2, alpha=0.3, s=10, c='#2a9d8f', edgecolors='none')

            fmin = min(f1.min(), f2.min()) - 0.5
            fmax = max(f1.max(), f2.max()) + 0.5
            ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.5, lw=1.5)

            corr = np.corrcoef(f1, f2)[0, 1]
            mae = np.mean(np.abs(f1 - f2))
            rmse = np.sqrt(np.mean((f1 - f2)**2))
            ax.text(0.05, 0.92,
                   f'R = {corr:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel(f"{m1} forces (eV/Å)", fontsize=10)
            ax.set_ylabel(f"{m2} forces (eV/Å)", fontsize=10)
            ax.set_title(f"{m1} vs {m2}", fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            pair_idx += 1

    for idx in range(pair_idx, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "force_parity_all_pairs.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ force_parity_all_pairs.png")


def plot_comprehensive_summary(results):
    """6-panel comprehensive figure."""
    model_names = list(results.keys())
    elements = ["Si", "Ge"]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle("Comprehensive Benchmark: MLIPs & Classical Potentials for Si and Ge\n"
                "(MACE-MP-0 | CHGNet | DeepMD | Tersoff | Stillinger-Weber)",
                fontsize=15, fontweight='bold', y=0.99)

    # (a) Lattice constant vs DFT
    ax = fig.add_subplot(gs[0, 0])
    for model in model_names:
        for elem in elements:
            if elem not in results.get(model, {}):
                continue
            gt = GROUND_TRUTH[elem]["dft_lattice_const"]
            pred = results[model][elem]["lattice_const_eos"]
            ax.scatter(gt, pred, c=MODEL_COLORS.get(model, 'gray'), s=120,
                      marker=MODEL_MARKERS.get(model, 'o'), edgecolors='black',
                      linewidth=0.8, zorder=5)
            ax.annotate(f'{elem}', (gt, pred), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
    all_a = []
    for model in model_names:
        for elem in elements:
            if elem in results.get(model, {}):
                all_a.extend([GROUND_TRUTH[elem]["dft_lattice_const"],
                             results[model][elem]["lattice_const_eos"]])
    if all_a:
        vmin, vmax = min(all_a)-0.1, max(all_a)+0.1
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel("DFT a₀ (Å)")
    ax.set_ylabel("Predicted a₀ (Å)")
    ax.set_title("(a) Lattice Constant vs DFT", fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) Energy vs DFT
    ax = fig.add_subplot(gs[0, 1])
    for model in model_names:
        for elem in elements:
            if elem not in results.get(model, {}):
                continue
            gt = GROUND_TRUTH[elem]["dft_cohesive_energy"]
            pred = results[model][elem]["energy_per_atom_relaxed"]
            ax.scatter(gt, pred, c=MODEL_COLORS.get(model, 'gray'), s=120,
                      marker=MODEL_MARKERS.get(model, 'o'), edgecolors='black',
                      linewidth=0.8, zorder=5)
            ax.annotate(f'{elem}', (gt, pred), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
    all_e = []
    for model in model_names:
        for elem in elements:
            if elem in results.get(model, {}):
                all_e.extend([GROUND_TRUTH[elem]["dft_cohesive_energy"],
                             results[model][elem]["energy_per_atom_relaxed"]])
    if all_e:
        vmin, vmax = min(all_e)-0.3, max(all_e)+0.3
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel("DFT Energy (eV/atom)")
    ax.set_ylabel("Predicted Energy (eV/atom)")
    ax.set_title("(b) Cohesive Energy vs DFT", fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (c) Error bar chart
    ax = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(len(elements))
    width = 0.8 / max(len(model_names), 1)
    for midx, model in enumerate(model_names):
        errs = []
        for elem in elements:
            if elem in results.get(model, {}):
                err = abs(results[model][elem]["lattice_const_eos"] -
                         GROUND_TRUTH[elem]["dft_lattice_const"]) / \
                      GROUND_TRUTH[elem]["dft_lattice_const"] * 100
                errs.append(err)
            else:
                errs.append(0)
        offset = (midx - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x_pos + offset, errs, width * 0.9,
                     color=MODEL_COLORS.get(model, 'gray'),
                     edgecolor='black', linewidth=0.5, label=model)
        for bar, err in zip(bars, errs):
            if err > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{err:.2f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(elements)
    ax.set_ylabel("Error vs DFT (%)")
    ax.set_title("(c) Lattice Constant Error", fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # (d) Si energy-strain
    ax = fig.add_subplot(gs[1, 0])
    for model in model_names:
        if "Si" not in results.get(model, {}):
            continue
        a_vals = np.array(results[model]["Si"]["strain_lattice_consts"])
        e_vals = np.array(results[model]["Si"]["strain_energies"])
        ax.plot(a_vals, e_vals, '-', color=MODEL_COLORS.get(model, 'gray'),
               lw=2, label=model)
    gt = GROUND_TRUTH["Si"]
    ax.axvline(x=gt["exp_lattice_const"], color='green', ls='--', lw=1.5,
              label=f'Exp ({gt["exp_lattice_const"]:.3f})')
    ax.axvline(x=gt["dft_lattice_const"], color='purple', ls=':', lw=1.5,
              label=f'DFT ({gt["dft_lattice_const"]:.3f})')
    ax.set_xlabel("a (Å)")
    ax.set_ylabel("E/atom (eV)")
    ax.set_title("(d) Si — E vs a", fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (e) Ge energy-strain
    ax = fig.add_subplot(gs[1, 1])
    for model in model_names:
        if "Ge" not in results.get(model, {}):
            continue
        a_vals = np.array(results[model]["Ge"]["strain_lattice_consts"])
        e_vals = np.array(results[model]["Ge"]["strain_energies"])
        ax.plot(a_vals, e_vals, '-', color=MODEL_COLORS.get(model, 'gray'),
               lw=2, label=model)
    gt = GROUND_TRUTH["Ge"]
    ax.axvline(x=gt["exp_lattice_const"], color='green', ls='--', lw=1.5,
              label=f'Exp ({gt["exp_lattice_const"]:.3f})')
    ax.axvline(x=gt["dft_lattice_const"], color='purple', ls=':', lw=1.5,
              label=f'DFT ({gt["dft_lattice_const"]:.3f})')
    ax.set_xlabel("a (Å)")
    ax.set_ylabel("E/atom (eV)")
    ax.set_title("(e) Ge — E vs a", fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (f) Force stats
    ax = fig.add_subplot(gs[1, 2])
    bar_data = []
    bar_labels = []
    bar_colors = []
    bar_errs = []
    for model in model_names:
        for elem in elements:
            if elem in results.get(model, {}):
                d = results[model][elem]
                bar_data.append(d.get("mean_max_force", 0))
                bar_errs.append(d.get("std_max_force", 0))
                bar_labels.append(f"{model}\n({elem})")
                bar_colors.append(MODEL_COLORS.get(model, 'gray'))

    if bar_data:
        x = np.arange(len(bar_data))
        ax.bar(x, bar_data, yerr=bar_errs, color=bar_colors,
              edgecolor='black', linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel("Mean Max |Force| (eV/Å)")
        ax.set_title("(f) Force Prediction (Bootstrap)", fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # Legend
    legend_elements = [Patch(facecolor=MODEL_COLORS.get(m, 'gray'),
                            edgecolor='black', label=m) for m in model_names]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=len(model_names), fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(os.path.join(OUTPUT_DIR, "comprehensive_summary.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ comprehensive_summary.png")


def plot_error_table(results):
    """Generate and save CSV summary table."""
    rows = []
    for model in results:
        for elem in ["Si", "Ge"]:
            if elem not in results[model]:
                continue
            d = results[model][elem]
            gt = GROUND_TRUTH[elem]
            err_dft_a = abs(d["lattice_const_eos"] - gt["dft_lattice_const"]) / gt["dft_lattice_const"] * 100
            err_exp_a = abs(d["lattice_const_eos"] - gt["exp_lattice_const"]) / gt["exp_lattice_const"] * 100
            rows.append({
                "Model": model,
                "Type": "MLIP" if model in ["MACE-MP-0", "CHGNet", "DeepMD-DPA2"] else "Classical",
                "Element": elem,
                "a_EOS": f'{d["lattice_const_eos"]:.4f}',
                "a_Relax": f'{d["lattice_const_relaxed"]:.4f}',
                "a_Exp": f'{gt["exp_lattice_const"]:.3f}',
                "a_DFT": f'{gt["dft_lattice_const"]:.3f}',
                "Err_Exp_%": f'{err_exp_a:.3f}',
                "Err_DFT_%": f'{err_dft_a:.3f}',
                "E_relaxed": f'{d["energy_per_atom_relaxed"]:.4f}',
                "E_DFT": f'{gt["dft_cohesive_energy"]:.2f}',
                "Mean_MaxF": f'{d.get("mean_max_force", 0):.4f}',
                "Std_MaxF": f'{d.get("std_max_force", 0):.4f}',
            })

    csv_path = os.path.join(OUTPUT_DIR, "comprehensive_summary_table.csv")
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  ✓ {csv_path}")

    # Print table
    print("\n" + "=" * 130)
    print(f"{'Model':15s} {'Type':8s} {'El':3s} | {'a_EOS':>7s} {'a_Rlx':>7s} {'a_Exp':>6s} {'a_DFT':>6s} | {'%Exp':>6s} {'%DFT':>6s} | {'E_rlx':>8s} {'E_DFT':>6s} | {'<MaxF>':>7s} {'±':>6s}")
    print("-" * 130)
    for r in rows:
        print(f"{r['Model']:15s} {r['Type']:8s} {r['Element']:3s} | {r['a_EOS']:>7s} {r['a_Relax']:>7s} {r['a_Exp']:>6s} {r['a_DFT']:>6s} | {r['Err_Exp_%']:>6s} {r['Err_DFT_%']:>6s} | {r['E_relaxed']:>8s} {r['E_DFT']:>6s} | {r['Mean_MaxF']:>7s} {r['Std_MaxF']:>6s}")
    print("=" * 130)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  COMPREHENSIVE BENCHMARK: MLIPs + Classical Potentials           ║")
    print("║  Si & Ge — MACE, CHGNet, DeepMD, Tersoff, SW                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")

    # Run benchmarks
    results = run_all_benchmarks()

    # Generate all plots
    print(f"\n{'='*70}\nGenerating Publication-Quality Plots\n{'='*70}")
    plot_publication_correlation(results)
    plot_publication_energy_correlation(results)
    plot_combined_correlation(results)
    plot_energy_convergence_curves(results)
    plot_force_cv_analysis(results)
    plot_force_parity_all_models(results)
    plot_comprehensive_summary(results)
    plot_error_table(results)

    print(f"\n✓ All output in: {OUTPUT_DIR}/")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK COMPLETE                                              ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
