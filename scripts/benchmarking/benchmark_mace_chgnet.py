#!/usr/bin/env python3
"""
Checkpoint 1 & 2: Benchmark pre-trained MACE and CHGNet MLIPs
against DFT/experimental ground truth for Si and Ge.

Properties benchmarked:
  - Lattice constant (Å)
  - Cohesive energy (eV/atom)
  - Internal forces (eV/Å) at equilibrium and strained configurations

Ground Truth Sources:
  - Experimental lattice constants: Si = 5.431 Å, Ge = 5.658 Å
  - DFT (PBE) lattice constants from Materials Project: Si ≈ 5.469 Å (mp-149), Ge ≈ 5.763 Å (mp-32)
  - Experimental cohesive energies: Si = -4.63 eV/atom, Ge = -3.85 eV/atom
  - DFT (PBE) cohesive energies: Si ≈ -5.43 eV/atom, Ge ≈ -4.62 eV/atom
"""

import os
import sys
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ─── ASE imports ───
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import ExpCellFilter, StrainFilter
from ase.eos import EquationOfState

OUTPUT_DIR = "/home/bib569/Workspace/Hackrush_2026-Problem-10/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Ground Truth / Reference Data
# ══════════════════════════════════════════════════════════════════════

GROUND_TRUTH = {
    "Si": {
        "exp_lattice_const": 5.431,       # Å, experimental
        "dft_lattice_const": 5.469,       # Å, Materials Project PBE (mp-149)
        "exp_cohesive_energy": -4.63,     # eV/atom, experimental
        "dft_cohesive_energy": -5.43,     # eV/atom, DFT PBE
    },
    "Ge": {
        "exp_lattice_const": 5.658,       # Å, experimental
        "dft_lattice_const": 5.763,       # Å, Materials Project PBE (mp-32)
        "exp_cohesive_energy": -3.85,     # eV/atom, experimental
        "dft_cohesive_energy": -4.62,     # eV/atom, DFT PBE
    }
}

# ══════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════

def get_lattice_constant_eos(atoms_func, calculator, element, n_points=7, strain_range=0.04):
    """
    Compute equilibrium lattice constant via Equation of State fitting.
    Creates multiple strained unit cells, calculates energy for each,
    and fits the Birch-Murnaghan EOS.
    """
    # Get a reference structure
    ref = atoms_func(element, crystalstructure='diamond', a=GROUND_TRUTH[element]["exp_lattice_const"])
    ref.calc = calculator
    
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    volumes = []
    energies = []
    lattice_consts = []
    
    strains = np.linspace(1 - strain_range, 1 + strain_range, n_points)
    
    for s in strains:
        a_test = a0 * s
        atoms = atoms_func(element, crystalstructure='diamond', a=a_test)
        atoms.calc = calculator
        e = atoms.get_potential_energy()
        v = atoms.get_volume()
        volumes.append(v)
        energies.append(e)
        lattice_consts.append(a_test)
    
    volumes = np.array(volumes)
    energies = np.array(energies)
    lattice_consts = np.array(lattice_consts)
    
    # Fit EOS
    try:
        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        v0, e0, B = eos.fit()
        # Convert v0 back to lattice constant for diamond cubic (8 atoms in conventional cell)
        n_atoms = len(atoms_func(element, crystalstructure='diamond', a=a0))
        a_eq = (v0 / (n_atoms / 8.0) * 4.0) ** (1.0/3.0)  # diamond cubic: V = a^3 / 4 per atom pair
        # Actually for diamond cubic with 2 atoms in primitive cell:
        # V_conv = a^3, with 8 atoms. V_prim = a^3/4 with 2 atoms.
        # ASE bulk with diamond gives 2 atoms (primitive cell), so V = a^3/4
        a_eq = (v0 * 4.0) ** (1.0/3.0)
    except Exception as e:
        print(f"  EOS fit failed for {element}: {e}")
        # Fallback: use minimum energy point
        idx = np.argmin(energies)
        a_eq = lattice_consts[idx]
        e0 = energies[idx] 
        v0 = volumes[idx]
    
    return a_eq, e0, volumes, energies, lattice_consts


def get_forces_at_configs(atoms_func, calculator, element, n_displaced=5):
    """
    Calculate forces at equilibrium and at several displaced configurations.
    Returns list of (displacement_magnitude, forces_array) tuples.
    """
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    results = []
    
    # Equilibrium configuration
    atoms_eq = atoms_func(element, crystalstructure='diamond', a=a0)
    atoms_eq.calc = calculator
    f_eq = atoms_eq.get_forces()
    results.append((0.0, f_eq.copy()))
    
    # Displaced configurations  
    np.random.seed(42)
    displacements = np.linspace(0.01, 0.15, n_displaced)
    
    for disp in displacements:
        atoms = atoms_func(element, crystalstructure='diamond', a=a0)
        # Apply random displacements
        pos = atoms.get_positions()
        delta = np.random.randn(*pos.shape) * disp
        atoms.set_positions(pos + delta)
        atoms.calc = calculator
        f = atoms.get_forces()
        results.append((disp, f.copy()))
    
    return results


def get_energy_vs_strain(atoms_func, calculator, element, n_points=15):
    """
    Calculate energy per atom as function of volumetric strain.
    """
    a0 = GROUND_TRUTH[element]["exp_lattice_const"]
    strains = np.linspace(0.85, 1.15, n_points)
    energies_per_atom = []
    strain_values = []
    
    for s in strains:
        a_test = a0 * s
        atoms = atoms_func(element, crystalstructure='diamond', a=a_test)
        atoms.calc = calculator
        e = atoms.get_potential_energy() / len(atoms)
        energies_per_atom.append(e)
        strain_values.append(s)
    
    return np.array(strain_values), np.array(energies_per_atom)


def relax_structure(atoms, calculator, fmax=0.01):
    """
    Relax atomic positions and cell shape/volume.
    Returns relaxed atoms with final energy and forces.
    """
    atoms.calc = calculator
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile=None)
    try:
        opt.run(fmax=fmax, steps=200)
    except Exception as e:
        print(f"  Relaxation warning: {e}")
    
    return atoms


# ══════════════════════════════════════════════════════════════════════
# Main Benchmark
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    results = {}
    
    # ─── Load MACE pre-trained model ───
    print("=" * 70)
    print("Loading pre-trained MACE-MP-0 model...")
    print("=" * 70)
    try:
        from mace.calculators import mace_mp
        mace_calc_fn = lambda: mace_mp(model="medium", dispersion=False, default_dtype="float32")
        mace_loaded = True
        print("  ✓ MACE-MP-0 loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load MACE: {e}")
        mace_loaded = False
    
    # ─── Load CHGNet pre-trained model ───
    print("\nLoading pre-trained CHGNet model...")
    try:
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        chgnet_model = CHGNet.load()
        chgnet_calc_fn = lambda: CHGNetCalculator(model=chgnet_model)
        chgnet_loaded = True
        print("  ✓ CHGNet loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load CHGNet: {e}")
        chgnet_loaded = False
    
    models = {}
    if mace_loaded:
        models["MACE-MP-0"] = mace_calc_fn
    if chgnet_loaded:
        models["CHGNet"] = chgnet_calc_fn
    
    if not models:
        print("\nERROR: No models could be loaded. Exiting.")
        sys.exit(1)
    
    # ─── Run benchmarks for each model and element ───
    for model_name, calc_fn in models.items():
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'=' * 70}")
        results[model_name] = {}
        
        for element in ["Si", "Ge"]:
            print(f"\n  --- {element} ---")
            gt = GROUND_TRUTH[element]
            try:
                calc = calc_fn()
                
                # 1. Lattice constant via EOS
                print(f"  Computing lattice constant (EOS fitting)...")
                a_eq, e_min, vols, ens, a_vals = get_lattice_constant_eos(
                    bulk, calc, element, n_points=11, strain_range=0.06
                )
                energy_per_atom = e_min / 2.0  # 2 atoms in primitive diamond cell
                
                print(f"    Predicted lattice constant:  {a_eq:.4f} Å")
                print(f"    Experimental:                {gt['exp_lattice_const']:.4f} Å")
                print(f"    DFT (PBE):                   {gt['dft_lattice_const']:.4f} Å")
                print(f"    Error vs Exp:                {abs(a_eq - gt['exp_lattice_const']):.4f} Å ({abs(a_eq - gt['exp_lattice_const'])/gt['exp_lattice_const']*100:.2f}%)")
                print(f"    Error vs DFT:                {abs(a_eq - gt['dft_lattice_const']):.4f} Å ({abs(a_eq - gt['dft_lattice_const'])/gt['dft_lattice_const']*100:.2f}%)")
                
                # 2. Energy per atom
                print(f"    Predicted energy/atom:        {energy_per_atom:.4f} eV")
                print(f"    Experimental cohesive E:      {gt['exp_cohesive_energy']:.4f} eV/atom")
                print(f"    DFT cohesive E:               {gt['dft_cohesive_energy']:.4f} eV/atom")
                
                # 3. Lattice constant via cell relaxation
                print(f"  Relaxing structure...")
                atoms_relax = bulk(element, crystalstructure='diamond', a=gt['exp_lattice_const'])
                calc_relax = calc_fn()
                atoms_relax = relax_structure(atoms_relax, calc_relax)
                cell = atoms_relax.get_cell()
                a_relaxed = np.mean([np.linalg.norm(cell[i]) for i in range(3)]) * (2**0.5)  # primitive to conventional
                e_relaxed = atoms_relax.get_potential_energy() / len(atoms_relax)
                f_relaxed = atoms_relax.get_forces()
                max_force_relaxed = np.max(np.abs(f_relaxed))
                
                print(f"    Relaxed lattice constant:     {a_relaxed:.4f} Å")
                print(f"    Relaxed energy/atom:           {e_relaxed:.4f} eV")
                print(f"    Max residual force:            {max_force_relaxed:.6f} eV/Å")
                
                # 4. Forces at displaced configurations
                print(f"  Computing forces at displaced configs...")
                calc_forces = calc_fn()
                force_results = get_forces_at_configs(bulk, calc_forces, element, n_displaced=8)
                
                # 5. Energy vs strain
                print(f"  Computing energy-strain curve...")
                calc_strain = calc_fn()
                strain_vals, e_strain = get_energy_vs_strain(bulk, calc_strain, element, n_points=21)
                
                results[model_name][element] = {
                    "lattice_const_eos": float(a_eq),
                    "lattice_const_relaxed": float(a_relaxed),
                    "energy_per_atom_eos": float(energy_per_atom),
                    "energy_per_atom_relaxed": float(e_relaxed),
                    "max_force_relaxed": float(max_force_relaxed),
                    "eos_volumes": vols.tolist(),
                    "eos_energies": ens.tolist(),
                    "eos_lattice_consts": a_vals.tolist(),
                    "force_results": [(float(d), f.tolist()) for d, f in force_results],
                    "strain_values": strain_vals.tolist(),
                    "energy_vs_strain": e_strain.tolist(),
                }
            except Exception as exc:
                import traceback
                print(f"  ✗ ERROR benchmarking {model_name}/{element}: {exc}")
                traceback.print_exc()
                continue
    
    # Save raw results
    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {OUTPUT_DIR}/benchmark_results.json")
    
    return results


# ══════════════════════════════════════════════════════════════════════
# Plotting Functions
# ══════════════════════════════════════════════════════════════════════

def create_parity_plots(results):
    """
    Create parity plots comparing predicted vs ground truth values.
    Checkpoint 2: The most important deliverable.
    """
    
    # ─── Color scheme ───
    colors = {
        "MACE-MP-0": "#E63946",   # Red
        "CHGNet": "#457B9D",       # Blue
    }
    markers = {
        "MACE-MP-0": "o",
        "CHGNet": "s",
    }
    
    # ══════════════════════════════════════════════════════════════
    # FIGURE 1: Lattice Constant Parity Plot
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Parity Plots: Predicted vs Ground Truth", fontsize=16, fontweight='bold', y=1.02)
    
    # --- Lattice Constant ---
    ax = axes[0]
    gt_lc_exp = []
    gt_lc_dft = []
    pred_lc = {m: [] for m in results}
    elements_order = []
    
    for element in ["Si", "Ge"]:
        gt = GROUND_TRUTH[element]
        gt_lc_exp.append(gt["exp_lattice_const"])
        gt_lc_dft.append(gt["dft_lattice_const"])
        elements_order.append(element)
        for model_name in results:
            pred_lc[model_name].append(results[model_name][element]["lattice_const_eos"])
    
    gt_lc_exp = np.array(gt_lc_exp)
    gt_lc_dft = np.array(gt_lc_dft)
    
    # Plot perfect parity line
    all_lc_vals = list(gt_lc_exp) + list(gt_lc_dft)
    for m in pred_lc:
        all_lc_vals.extend(pred_lc[m])
    lc_min, lc_max = min(all_lc_vals) - 0.1, max(all_lc_vals) + 0.1
    ax.plot([lc_min, lc_max], [lc_min, lc_max], 'k--', alpha=0.5, lw=1.5, label='Perfect parity')
    ax.fill_between([lc_min, lc_max], [lc_min - 0.05, lc_max - 0.05], [lc_min + 0.05, lc_max + 0.05], 
                    alpha=0.1, color='gray', label='±0.05 Å band')
    
    for model_name in results:
        pred = np.array(pred_lc[model_name])
        # vs Experimental
        ax.scatter(gt_lc_exp, pred, c=colors.get(model_name, 'green'), 
                  marker=markers.get(model_name, 'D'), s=120, edgecolors='black', linewidth=0.8,
                  label=f'{model_name} vs Exp', zorder=5)
        # vs DFT
        ax.scatter(gt_lc_dft, pred, c=colors.get(model_name, 'green'),
                  marker=markers.get(model_name, 'D'), s=120, edgecolors='black', linewidth=0.8,
                  facecolors='none', label=f'{model_name} vs DFT', zorder=5)
    
    # Annotate points
    for i, elem in enumerate(elements_order):
        for model_name in results:
            pred_val = pred_lc[model_name][i]
            ax.annotate(elem, (gt_lc_exp[i], pred_val), textcoords="offset points", 
                       xytext=(8, 5), fontsize=9, fontstyle='italic')
    
    ax.set_xlabel("Ground Truth Lattice Constant (Å)", fontsize=12)
    ax.set_ylabel("Predicted Lattice Constant (Å)", fontsize=12)
    ax.set_title("Lattice Constant", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(lc_min, lc_max)
    ax.set_ylim(lc_min, lc_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # --- Cohesive Energy ---
    ax = axes[1]
    gt_ce_exp = []
    gt_ce_dft = []
    pred_ce = {m: [] for m in results}
    
    for element in ["Si", "Ge"]:
        gt = GROUND_TRUTH[element]
        gt_ce_exp.append(gt["exp_cohesive_energy"])
        gt_ce_dft.append(gt["dft_cohesive_energy"])
        for model_name in results:
            pred_ce[model_name].append(results[model_name][element]["energy_per_atom_relaxed"])
    
    gt_ce_exp = np.array(gt_ce_exp)
    gt_ce_dft = np.array(gt_ce_dft)
    
    all_ce_vals = list(gt_ce_exp) + list(gt_ce_dft)
    for m in pred_ce:
        all_ce_vals.extend(pred_ce[m])
    ce_min, ce_max = min(all_ce_vals) - 0.3, max(all_ce_vals) + 0.3
    ax.plot([ce_min, ce_max], [ce_min, ce_max], 'k--', alpha=0.5, lw=1.5, label='Perfect parity')
    ax.fill_between([ce_min, ce_max], [ce_min - 0.2, ce_max - 0.2], [ce_min + 0.2, ce_max + 0.2],
                    alpha=0.1, color='gray', label='±0.2 eV band')
    
    for model_name in results:
        pred = np.array(pred_ce[model_name])
        ax.scatter(gt_ce_exp, pred, c=colors.get(model_name, 'green'),
                  marker=markers.get(model_name, 'D'), s=120, edgecolors='black', linewidth=0.8,
                  label=f'{model_name} vs Exp', zorder=5)
        ax.scatter(gt_ce_dft, pred, c=colors.get(model_name, 'green'),
                  marker=markers.get(model_name, 'D'), s=120, edgecolors='black', linewidth=0.8,
                  facecolors='none', label=f'{model_name} vs DFT', zorder=5)
    
    for i, elem in enumerate(elements_order):
        for model_name in results:
            pred_val = pred_ce[model_name][i]
            ax.annotate(elem, (gt_ce_exp[i], pred_val), textcoords="offset points",
                       xytext=(8, 5), fontsize=9, fontstyle='italic')
    
    ax.set_xlabel("Ground Truth Energy (eV/atom)", fontsize=12)
    ax.set_ylabel("Predicted Energy (eV/atom)", fontsize=12)
    ax.set_title("Cohesive Energy per Atom", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(ce_min, ce_max)
    ax.set_ylim(ce_min, ce_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "parity_plot_lattice_energy.png"), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved parity_plot_lattice_energy.png")
    plt.close()
    
    # ══════════════════════════════════════════════════════════════
    # FIGURE 2: Force Parity Plot  
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Force Parity: Predicted Forces at Displaced Configurations",
                fontsize=14, fontweight='bold', y=1.02)
    
    for idx, element in enumerate(["Si", "Ge"]):
        ax = axes[idx]
        
        for model_name in results:
            force_data = results[model_name][element]["force_results"]
            # At equilibrium, forces should be ~0 (ground truth = 0)
            # At displaced configs, we compare force magnitudes
            # Since we don't have DFT forces for arbitrary displacements,
            # we'll show force magnitude vs displacement (demonstrates restoring forces)
            disps = []
            max_forces = []
            mean_forces = []
            
            for disp, forces in force_data:
                forces_arr = np.array(forces)
                disps.append(disp)
                max_forces.append(np.max(np.linalg.norm(forces_arr, axis=1)))
                mean_forces.append(np.mean(np.linalg.norm(forces_arr, axis=1)))
            
            ax.plot(disps, max_forces, '-o', color=colors.get(model_name, 'green'),
                   markersize=7, linewidth=2, label=f'{model_name} (max |F|)',
                   markeredgecolor='black', markeredgewidth=0.5)
            ax.plot(disps, mean_forces, '--s', color=colors.get(model_name, 'green'),
                   markersize=6, linewidth=1.5, alpha=0.7, label=f'{model_name} (mean |F|)',
                   markeredgecolor='black', markeredgewidth=0.5)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel("Displacement Magnitude (Å)", fontsize=12)
        ax.set_ylabel("Force Magnitude (eV/Å)", fontsize=12)
        ax.set_title(f"{element} - Forces vs Displacement", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "force_vs_displacement.png"), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved force_vs_displacement.png")
    plt.close()
    
    # ══════════════════════════════════════════════════════════════
    # FIGURE 3: Energy-Strain Curves
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Energy-Volume Curves: MLIP Predictions vs Ground Truth Minima",
                fontsize=14, fontweight='bold', y=1.02)
    
    for idx, element in enumerate(["Si", "Ge"]):
        ax = axes[idx]
        gt = GROUND_TRUTH[element]
        
        for model_name in results:
            strain_vals = np.array(results[model_name][element]["strain_values"])
            e_strain = np.array(results[model_name][element]["energy_vs_strain"])
            
            ax.plot(strain_vals * gt["exp_lattice_const"], e_strain, '-', 
                   color=colors.get(model_name, 'green'), linewidth=2.5,
                   label=f'{model_name}')
        
        # Mark ground truth positions
        ax.axvline(x=gt["exp_lattice_const"], color='green', linestyle='--', alpha=0.7, 
                  linewidth=1.5, label=f'Exp a₀ = {gt["exp_lattice_const"]:.3f} Å')
        ax.axvline(x=gt["dft_lattice_const"], color='purple', linestyle=':', alpha=0.7,
                  linewidth=1.5, label=f'DFT a₀ = {gt["dft_lattice_const"]:.3f} Å')
        
        ax.set_xlabel("Lattice Constant (Å)", fontsize=12)
        ax.set_ylabel("Energy per Atom (eV)", fontsize=12)
        ax.set_title(f"{element} - Energy vs Lattice Constant", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "energy_strain_curves.png"), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved energy_strain_curves.png")
    plt.close()
    
    # ══════════════════════════════════════════════════════════════
    # FIGURE 4: Comprehensive Force Parity Plot (MLIP vs MLIP)
    # ══════════════════════════════════════════════════════════════
    model_names = list(results.keys())
    if len(model_names) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Force Parity: MACE-MP-0 vs CHGNet Predicted Forces",
                    fontsize=14, fontweight='bold', y=1.02)
        
        for idx, element in enumerate(["Si", "Ge"]):
            ax = axes[idx]
            
            # Collect all force components from both models at same displacements
            m1, m2 = model_names[0], model_names[1]
            f1_all = []
            f2_all = []
            
            force_data_1 = results[m1][element]["force_results"]
            force_data_2 = results[m2][element]["force_results"]
            
            n_configs = min(len(force_data_1), len(force_data_2))
            for i in range(n_configs):
                f1 = np.array(force_data_1[i][1]).flatten()
                f2 = np.array(force_data_2[i][1]).flatten()
                f1_all.extend(f1.tolist())
                f2_all.extend(f2.tolist())
            
            f1_all = np.array(f1_all)
            f2_all = np.array(f2_all)
            
            ax.scatter(f1_all, f2_all, alpha=0.4, s=20, c='#2a9d8f', edgecolors='none')
            fmin = min(f1_all.min(), f2_all.min()) - 0.5
            fmax = max(f1_all.max(), f2_all.max()) + 0.5
            ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.5, lw=1.5, label='Perfect parity')
            
            # Compute correlation
            corr = np.corrcoef(f1_all, f2_all)[0, 1]
            mae = np.mean(np.abs(f1_all - f2_all))
            ax.text(0.05, 0.92, f'R = {corr:.4f}\nMAE = {mae:.4f} eV/Å',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel(f"{m1} Forces (eV/Å)", fontsize=12)
            ax.set_ylabel(f"{m2} Forces (eV/Å)", fontsize=12)
            ax.set_title(f"{element}", fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "force_parity_mace_vs_chgnet.png"), dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved force_parity_mace_vs_chgnet.png")
        plt.close()


def create_summary_table(results):
    """
    Create a comprehensive comparison table.
    """
    print("\n" + "=" * 100)
    print("SUMMARY: Predicted vs Ground Truth")
    print("=" * 100)
    
    header = f"{'Model':<15} {'Element':<8} {'a(EOS) Å':<12} {'a(Relax) Å':<12} {'Exp a Å':<10} {'DFT a Å':<10} {'ΔExp %':<9} {'ΔDFT %':<9} {'E/atom eV':<12} {'Exp E eV':<10} {'DFT E eV':<10}"
    print(header)
    print("-" * 100)
    
    rows = []
    for model_name in results:
        for element in ["Si", "Ge"]:
            r = results[model_name][element]
            gt = GROUND_TRUTH[element]
            
            err_exp = abs(r["lattice_const_eos"] - gt["exp_lattice_const"]) / gt["exp_lattice_const"] * 100
            err_dft = abs(r["lattice_const_eos"] - gt["dft_lattice_const"]) / gt["dft_lattice_const"] * 100
            
            row = f"{model_name:<15} {element:<8} {r['lattice_const_eos']:<12.4f} {r['lattice_const_relaxed']:<12.4f} {gt['exp_lattice_const']:<10.3f} {gt['dft_lattice_const']:<10.3f} {err_exp:<9.2f} {err_dft:<9.2f} {r['energy_per_atom_relaxed']:<12.4f} {gt['exp_cohesive_energy']:<10.2f} {gt['dft_cohesive_energy']:<10.2f}"
            print(row)
            rows.append({
                "Model": model_name,
                "Element": element,
                "Lattice_EOS": r["lattice_const_eos"],
                "Lattice_Relaxed": r["lattice_const_relaxed"],
                "Exp_Lattice": gt["exp_lattice_const"],
                "DFT_Lattice": gt["dft_lattice_const"],
                "Error_Exp_pct": err_exp,
                "Error_DFT_pct": err_dft,
                "Energy_per_atom": r["energy_per_atom_relaxed"],
                "Exp_Energy": gt["exp_cohesive_energy"],
                "DFT_Energy": gt["dft_cohesive_energy"],
                "Max_Force_Relaxed": r["max_force_relaxed"],
            })
    
    print("=" * 100)
    
    # Save table as CSV
    import csv
    with open(os.path.join(OUTPUT_DIR, "summary_table.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Summary table saved to {OUTPUT_DIR}/summary_table.csv")
    
    return rows


# ══════════════════════════════════════════════════════════════════════
# COMPREHENSIVE PARITY PLOT  (Checkpoint 2 - Most Important)
# ══════════════════════════════════════════════════════════════════════

def create_comprehensive_parity(results):
    """
    Single comprehensive figure with all parity comparisons.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle("Comprehensive Parity Plots: Pre-trained MLIP Predictions vs Ground Truth\n(Si & Ge — Diamond Cubic)",
                fontsize=16, fontweight='bold', y=0.98)
    
    colors_model = {
        "MACE-MP-0": "#E63946",
        "CHGNet": "#457B9D",
    }
    
    # ── Panel (a): Lattice Constant vs Experimental ──
    ax = fig.add_subplot(gs[0, 0])
    all_gt, all_pred, all_labels = [], [], []
    for model_name in results:
        for element in ["Si", "Ge"]:
            gt_val = GROUND_TRUTH[element]["exp_lattice_const"]
            pred_val = results[model_name][element]["lattice_const_eos"]
            all_gt.append(gt_val)
            all_pred.append(pred_val)
            ax.scatter(gt_val, pred_val, c=colors_model.get(model_name, 'gray'),
                      s=150, marker='o', edgecolors='black', linewidth=0.8, zorder=5)
            ax.annotate(f'{element}\n({model_name.split("-")[0]})', (gt_val, pred_val),
                       textcoords="offset points", xytext=(10, -5), fontsize=7)
    
    vmin, vmax = min(all_gt + all_pred) - 0.1, max(all_gt + all_pred) + 0.1
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel("Experimental a (Å)", fontsize=11)
    ax.set_ylabel("Predicted a (Å)", fontsize=11)
    ax.set_title("(a) Lattice Const. vs Experimental", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ── Panel (b): Lattice Constant vs DFT ──
    ax = fig.add_subplot(gs[0, 1])
    all_gt, all_pred = [], []
    for model_name in results:
        for element in ["Si", "Ge"]:
            gt_val = GROUND_TRUTH[element]["dft_lattice_const"]
            pred_val = results[model_name][element]["lattice_const_eos"]
            all_gt.append(gt_val)
            all_pred.append(pred_val)
            ax.scatter(gt_val, pred_val, c=colors_model.get(model_name, 'gray'),
                      s=150, marker='o', edgecolors='black', linewidth=0.8, zorder=5)
            ax.annotate(f'{element}\n({model_name.split("-")[0]})', (gt_val, pred_val),
                       textcoords="offset points", xytext=(10, -5), fontsize=7)
    
    vmin, vmax = min(all_gt + all_pred) - 0.1, max(all_gt + all_pred) + 0.1
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel("DFT (PBE) a (Å)", fontsize=11)
    ax.set_ylabel("Predicted a (Å)", fontsize=11)
    ax.set_title("(b) Lattice Const. vs DFT", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ── Panel (c): Energy vs DFT ──
    ax = fig.add_subplot(gs[0, 2])
    all_gt, all_pred = [], []
    for model_name in results:
        for element in ["Si", "Ge"]:
            gt_val = GROUND_TRUTH[element]["dft_cohesive_energy"]
            pred_val = results[model_name][element]["energy_per_atom_relaxed"]
            all_gt.append(gt_val)
            all_pred.append(pred_val)
            ax.scatter(gt_val, pred_val, c=colors_model.get(model_name, 'gray'),
                      s=150, marker='o', edgecolors='black', linewidth=0.8, zorder=5)
            ax.annotate(f'{element}\n({model_name.split("-")[0]})', (gt_val, pred_val),
                       textcoords="offset points", xytext=(10, -5), fontsize=7)
    
    vmin, vmax = min(all_gt + all_pred) - 0.3, max(all_gt + all_pred) + 0.3
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, lw=1.5)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel("DFT Energy (eV/atom)", fontsize=11)
    ax.set_ylabel("Predicted Energy (eV/atom)", fontsize=11)
    ax.set_title("(c) Energy vs DFT", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ── Panel (d): Energy-Strain for Si ──
    ax = fig.add_subplot(gs[1, 0])
    for model_name in results:
        sv = np.array(results[model_name]["Si"]["strain_values"])
        ev = np.array(results[model_name]["Si"]["energy_vs_strain"])
        a_vals = sv * GROUND_TRUTH["Si"]["exp_lattice_const"]
        ax.plot(a_vals, ev, '-', color=colors_model.get(model_name, 'gray'),
               linewidth=2.5, label=model_name)
    ax.axvline(x=GROUND_TRUTH["Si"]["exp_lattice_const"], color='green', linestyle='--',
              alpha=0.7, lw=1.5, label=f'Exp ({GROUND_TRUTH["Si"]["exp_lattice_const"]:.3f} Å)')
    ax.axvline(x=GROUND_TRUTH["Si"]["dft_lattice_const"], color='purple', linestyle=':',
              alpha=0.7, lw=1.5, label=f'DFT ({GROUND_TRUTH["Si"]["dft_lattice_const"]:.3f} Å)')
    ax.set_xlabel("Lattice Constant (Å)", fontsize=11)
    ax.set_ylabel("Energy/atom (eV)", fontsize=11)
    ax.set_title("(d) Si — Energy vs Lattice Const.", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ── Panel (e): Energy-Strain for Ge ──
    ax = fig.add_subplot(gs[1, 1])
    for model_name in results:
        sv = np.array(results[model_name]["Ge"]["strain_values"])
        ev = np.array(results[model_name]["Ge"]["energy_vs_strain"])
        a_vals = sv * GROUND_TRUTH["Ge"]["exp_lattice_const"]
        ax.plot(a_vals, ev, '-', color=colors_model.get(model_name, 'gray'),
               linewidth=2.5, label=model_name)
    ax.axvline(x=GROUND_TRUTH["Ge"]["exp_lattice_const"], color='green', linestyle='--',
              alpha=0.7, lw=1.5, label=f'Exp ({GROUND_TRUTH["Ge"]["exp_lattice_const"]:.3f} Å)')
    ax.axvline(x=GROUND_TRUTH["Ge"]["dft_lattice_const"], color='purple', linestyle=':',
              alpha=0.7, lw=1.5, label=f'DFT ({GROUND_TRUTH["Ge"]["dft_lattice_const"]:.3f} Å)')
    ax.set_xlabel("Lattice Constant (Å)", fontsize=11)
    ax.set_ylabel("Energy/atom (eV)", fontsize=11)
    ax.set_title("(e) Ge — Energy vs Lattice Const.", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ── Panel (f): Force comparison ──
    ax = fig.add_subplot(gs[1, 2])
    model_names = list(results.keys())
    
    if len(model_names) >= 2:
        f1_all, f2_all = [], []
        for element in ["Si", "Ge"]:
            fd1 = results[model_names[0]][element]["force_results"]
            fd2 = results[model_names[1]][element]["force_results"]
            n = min(len(fd1), len(fd2))
            for i in range(n):
                f1_all.extend(np.array(fd1[i][1]).flatten().tolist())
                f2_all.extend(np.array(fd2[i][1]).flatten().tolist())
        
        f1_all = np.array(f1_all)
        f2_all = np.array(f2_all)
        ax.scatter(f1_all, f2_all, alpha=0.3, s=15, c='#2a9d8f', edgecolors='none')
        fmin = min(f1_all.min(), f2_all.min()) - 0.5
        fmax = max(f1_all.max(), f2_all.max()) + 0.5
        ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.5, lw=1.5)
        
        corr = np.corrcoef(f1_all, f2_all)[0, 1]
        mae = np.mean(np.abs(f1_all - f2_all))
        rmse = np.sqrt(np.mean((f1_all - f2_all)**2))
        ax.text(0.05, 0.92, f'R = {corr:.4f}\nMAE = {mae:.4f} eV/Å\nRMSE = {rmse:.4f} eV/Å',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(f"{model_names[0]} Forces (eV/Å)", fontsize=11)
        ax.set_ylabel(f"{model_names[1]} Forces (eV/Å)", fontsize=11)
        ax.set_title("(f) Force Parity (Si+Ge combined)", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
    else:
        # Single model: show force magnitudes
        for element in ["Si", "Ge"]:
            fd = results[model_names[0]][element]["force_results"]
            disps = [d for d, _ in fd]
            max_f = [np.max(np.linalg.norm(np.array(f), axis=1)) for _, f in fd]
            ax.plot(disps, max_f, '-o', label=f'{element}', markersize=6)
        ax.set_xlabel("Displacement (Å)", fontsize=11)
        ax.set_ylabel("Max |Force| (eV/Å)", fontsize=11)
        ax.set_title("(f) Force Response", fontsize=12, fontweight='bold')
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    # Add legend for model colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_model.get(m, 'gray'), edgecolor='black', label=m) 
                      for m in results]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(results), fontsize=12,
              bbox_to_anchor=(0.5, -0.02))
    
    fig.savefig(os.path.join(OUTPUT_DIR, "comprehensive_parity_plot.png"), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved comprehensive_parity_plot.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Error Bar Comparison Plot
# ══════════════════════════════════════════════════════════════════════

def create_error_comparison(results):
    """Bar chart showing % errors for each model/element."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Prediction Errors: MLIP vs Ground Truth", fontsize=15, fontweight='bold')
    
    model_names = list(results.keys())
    elements = ["Si", "Ge"]
    x = np.arange(len(elements))
    width = 0.35
    colors_model = {"MACE-MP-0": "#E63946", "CHGNet": "#457B9D"}
    
    # Lattice constant errors
    ax = axes[0]
    for i, model in enumerate(model_names):
        errs_exp = [abs(results[model][e]["lattice_const_eos"] - GROUND_TRUTH[e]["exp_lattice_const"]) 
                   / GROUND_TRUTH[e]["exp_lattice_const"] * 100 for e in elements]
        errs_dft = [abs(results[model][e]["lattice_const_eos"] - GROUND_TRUTH[e]["dft_lattice_const"])
                   / GROUND_TRUTH[e]["dft_lattice_const"] * 100 for e in elements]
        
        offset = (i - 0.5 * (len(model_names) - 1)) * width
        bars1 = ax.bar(x + offset - width/4, errs_exp, width/2, label=f'{model} vs Exp',
                       color=colors_model.get(model, 'gray'), alpha=0.9, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + offset + width/4, errs_dft, width/2, label=f'{model} vs DFT',
                       color=colors_model.get(model, 'gray'), alpha=0.5, edgecolor='black', linewidth=0.5,
                       hatch='///')
        
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Element", fontsize=12)
    ax.set_ylabel("Error (%)", fontsize=12)
    ax.set_title("Lattice Constant Error", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Energy comparison (absolute values side by side)
    ax = axes[1]
    for i, model in enumerate(model_names):
        pred_e = [results[model][e]["energy_per_atom_relaxed"] for e in elements]
        offset = (i - 0.5 * (len(model_names) - 1)) * width
        bars = ax.bar(x + offset, pred_e, width * 0.8, label=model,
                     color=colors_model.get(model, 'gray'), edgecolor='black', linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
                   f'{bar.get_height():.3f}', ha='center', va='top', fontsize=8, color='white')
    
    # Add ground truth markers
    for j, elem in enumerate(elements):
        ax.scatter(j, GROUND_TRUTH[elem]["exp_cohesive_energy"], marker='*', s=200,
                  c='lime', edgecolors='black', zorder=10, label=f'Exp {elem}' if j == 0 else '')
        ax.scatter(j, GROUND_TRUTH[elem]["dft_cohesive_energy"], marker='P', s=150,
                  c='purple', edgecolors='black', zorder=10, label=f'DFT {elem}' if j == 0 else '')
    
    ax.set_xlabel("Element", fontsize=12)
    ax.set_ylabel("Energy per Atom (eV)", fontsize=12)
    ax.set_title("Energy per Atom Comparison", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "error_comparison.png"), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved error_comparison.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Checkpoint 1 & 2: MLIP Benchmark for Si and Ge             ║")
    print("║  Pre-trained Models: MACE-MP-0, CHGNet                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    # Run benchmarks
    results = run_benchmark()
    
    # Create plots (Checkpoint 2)
    print("\n" + "=" * 70)
    print("Generating Parity Plots (Checkpoint 2)")
    print("=" * 70)
    
    create_parity_plots(results)
    create_comprehensive_parity(results)
    create_error_comparison(results)
    
    # Print summary table
    rows = create_summary_table(results)
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  All checkpoints complete!                                   ║")
    print(f"║  Results saved to: {OUTPUT_DIR:<41} ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
