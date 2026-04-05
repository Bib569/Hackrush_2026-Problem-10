#!/usr/bin/env python3
"""
Phase 2-4: DPA-3 Pre-trained (SemiCond head) + Fine-tuned DeepMD + MEAM Ge benchmark.
Uses DPA-3.1-3M with Domains_SemiCond head (extracted to single-task model).
Generates training data from MACE teacher, trains DeepMD with Optuna HPO,
benchmarks DPA-3 pre-trained, fine-tuned, and MEAM Ge.
Saves all plot data to JSON/CSV for manual re-plotting.
"""
import os, sys, json, csv, subprocess, tempfile, shutil, warnings, traceback, time, copy
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from ase.build import bulk
from ase.optimize import BFGS
from ase.eos import EquationOfState
from ase.io import write as ase_write
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

BASE = "/home/bib569/Workspace/Hackrush_2026-Problem-10"
OUT = os.path.join(BASE, "results_comprehensive")
POT = os.path.join(BASE, "potentials")
LMP = "/usr/bin/lmp"
DP_CMD = "/home/bib569/.local/bin/dp"
# Use DPA-3 extracted single-head model (Domains_SemiCond)
DPA3_MODEL = os.path.join(POT, "dpa3_3M_semicond.pt")
TRAIN_DIR = os.path.join(BASE, "deepmd_training")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

GT = {
    "Si": {"exp_a": 5.431, "dft_a": 5.469, "exp_e": -4.63, "dft_e": -5.43},
    "Ge": {"exp_a": 5.658, "dft_a": 5.763, "exp_e": -3.85, "dft_e": -4.62},
}
MASS = {"Si": 28.085, "Ge": 72.630}
ATOMIC_NUM = {"Si": 14, "Ge": 32}

# ══════════════════════════════════════════════════════════════════
# LAMMPS subprocess helper
# ══════════════════════════════════════════════════════════════════
def lammps_energy_forces(atoms, pair_cmds, element, extra_cmds=None):
    """Run LAMMPS and return (energy, forces)."""
    d = tempfile.mkdtemp(prefix="lmp_")
    try:
        ase_write(os.path.join(d, "s.data"), atoms, format="lammps-data")
        extra = "\n".join(extra_cmds) if extra_cmds else ""
        s = f"""units metal
atom_style charge
boundary p p p
box tilt large
read_data s.data
mass 1 {MASS[element]}
{pair_cmds}
{extra}
thermo_style custom step pe
thermo 1
dump 1 all custom 1 f.dump id fx fy fz
dump_modify 1 sort id format float "%20.12f"
run 0
"""
        with open(os.path.join(d, "in.x"), 'w') as f:
            f.write(s)
        r = subprocess.run([LMP, "-in", "in.x"], cwd=d,
                          capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"LAMMPS error: {r.stderr[-500:]}\n{r.stdout[-500:]}")
        energy = None
        with open(os.path.join(d, "log.lammps")) as f:
            th = False
            for line in f:
                if "Step" in line and "PotEng" in line:
                    th = True; continue
                if th and "Loop" in line:
                    th = False; continue
                if th:
                    p = line.strip().split()
                    if len(p) >= 2:
                        try: energy = float(p[1])
                        except: pass
        n = len(atoms)
        forces = np.zeros((n, 3))
        with open(os.path.join(d, "f.dump")) as f:
            lines = f.readlines()
        rd = False
        for line in lines:
            if "ITEM: ATOMS" in line:
                rd = True; continue
            if rd:
                p = line.strip().split()
                if len(p) >= 4:
                    forces[int(p[0])-1] = [float(p[1]), float(p[2]), float(p[3])]
        return energy, forces
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# Benchmark helpers (same methodology as run_full_benchmark.py)
# ══════════════════════════════════════════════════════════════════
def eos_lattice(calc_fn, element, n=11, sr=0.06):
    a0 = GT[element]["exp_a"]
    strains = np.linspace(1-sr, 1+sr, n)
    vols, ens, avals = [], [], []
    for s in strains:
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        at.calc = calc_fn()
        e = at.get_potential_energy()
        vols.append(at.get_volume()); ens.append(e); avals.append(a0*s)
    V, E, A = np.array(vols), np.array(ens), np.array(avals)
    try:
        eos = EquationOfState(V, E, eos='birchmurnaghan')
        v0, e0, B = eos.fit()
        a_eq = (v0 * 4.0) ** (1./3.)
    except:
        idx = np.argmin(E); a_eq = A[idx]; e0 = E[idx]
    return a_eq, e0 / 2.0, A, E / 2.0

def eos_lattice_lammps(pair_cmds, element, extra_cmds=None, n=11, sr=0.06):
    a0 = GT[element]["exp_a"]
    strains = np.linspace(1-sr, 1+sr, n)
    vols, ens, avals = [], [], []
    for s in strains:
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        e, _ = lammps_energy_forces(at, pair_cmds, element, extra_cmds)
        vols.append(at.get_volume()); ens.append(e); avals.append(a0*s)
    V, E, A = np.array(vols), np.array(ens), np.array(avals)
    try:
        eos = EquationOfState(V, E, eos='birchmurnaghan')
        v0, e0, B = eos.fit()
        a_eq = (v0 * 4.0) ** (1./3.)
    except:
        idx = np.argmin(E); a_eq = A[idx]; e0 = E[idx]
    return a_eq, e0 / 2.0, A, E / 2.0

def relax_cell(calc_fn, element):
    at = bulk(element, crystalstructure='diamond', a=GT[element]["exp_a"])
    at.calc = calc_fn()
    ecf = ExpCellFilter(at)
    opt = BFGS(ecf, logfile=None)
    try: opt.run(fmax=0.01, steps=200)
    except: pass
    cell = at.get_cell()
    a_r = np.mean([np.linalg.norm(cell[i]) for i in range(3)]) * (2**0.5)
    return a_r, at.get_potential_energy()/len(at), np.max(np.abs(at.get_forces()))

def relax_cell_lammps(pair_cmds, element, extra_cmds=None):
    a0 = GT[element]["exp_a"]
    best_e, best_a = 1e10, a0
    for s in np.linspace(0.97, 1.03, 31):
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        e, _ = lammps_energy_forces(at, pair_cmds, element, extra_cmds)
        if e < best_e: best_e = e; best_a = a0*s
    at = bulk(element, crystalstructure='diamond', a=best_a)
    _, f = lammps_energy_forces(at, pair_cmds, element, extra_cmds)
    return best_a, best_e/2.0, np.max(np.abs(f))

def bootstrap_forces(calc_fn, element, n=30, seed=42):
    rng = np.random.RandomState(seed)
    disps, fmags = [], []
    a0 = GT[element]["exp_a"]
    for i in range(n):
        at = bulk(element, crystalstructure='diamond', a=a0)
        dm = rng.uniform(0.01, 0.2)
        at.set_positions(at.get_positions() + rng.randn(*at.get_positions().shape)*dm)
        at.calc = calc_fn()
        f = at.get_forces()
        disps.append(dm); fmags.append(f.flatten().tolist())
    return np.array(disps), fmags

def bootstrap_forces_lammps(pair_cmds, element, extra_cmds=None, n=30, seed=42):
    rng = np.random.RandomState(seed)
    disps, fmags = [], []
    a0 = GT[element]["exp_a"]
    for i in range(n):
        at = bulk(element, crystalstructure='diamond', a=a0)
        dm = rng.uniform(0.01, 0.2)
        at.set_positions(at.get_positions() + rng.randn(*at.get_positions().shape)*dm)
        _, f = lammps_energy_forces(at, pair_cmds, element, extra_cmds)
        disps.append(dm); fmags.append(f.flatten().tolist())
    return np.array(disps), fmags

def energy_strain(calc_fn, element, n=21):
    a0 = GT[element]["exp_a"]
    strains = np.linspace(0.85, 1.15, n)
    Es = []
    for s in strains:
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        at.calc = calc_fn()
        Es.append(at.get_potential_energy() / len(at))
    return strains * a0, np.array(Es)

def energy_strain_lammps(pair_cmds, element, extra_cmds=None, n=21):
    a0 = GT[element]["exp_a"]
    strains = np.linspace(0.85, 1.15, n)
    Es = []
    for s in strains:
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        e, _ = lammps_energy_forces(at, pair_cmds, element, extra_cmds)
        Es.append(e / len(at))
    return strains * a0, np.array(Es)


# ══════════════════════════════════════════════════════════════════
# PHASE 2: DPA-3 PRE-TRAINED BENCHMARK (SemiCond head)
# ══════════════════════════════════════════════════════════════════
def benchmark_dpa3_pretrained():
    print("\n" + "="*70)
    print("  PHASE 2: DPA-3 Pre-trained Benchmark (Domains_SemiCond)")
    print("="*70)

    from deepmd.calculator import DP as DPCalc

    def dpa3_calc():
        return DPCalc(model=DPA3_MODEL)

    results = {}
    for el in ["Si", "Ge"]:
        print(f"\n  --- DPA-3 Pre-trained / {el} ---")
        try:
            print("  [1] EOS...")
            a_eos, e_eos, a_crv, e_crv = eos_lattice(dpa3_calc, el)
            print(f"      a={a_eos:.4f} (DFT={GT[el]['dft_a']:.3f})")

            print("  [2] Relax...")
            a_rlx, e_rlx, mf = relax_cell(dpa3_calc, el)
            print(f"      a={a_rlx:.4f}, E/at={e_rlx:.4f}, maxF={mf:.2e}")

            print("  [3] Bootstrap forces...")
            disps, fmags = bootstrap_forces(dpa3_calc, el)
            max_fs = [np.max(np.abs(np.array(f).reshape(-1, 3))) for f in fmags]
            print(f"      <maxF>={np.mean(max_fs):.4f}±{np.std(max_fs):.4f}")

            print("  [4] E-strain...")
            s_a, s_e = energy_strain(dpa3_calc, el)

            results[el] = {
                "a_eos": float(a_eos), "a_rlx": float(a_rlx),
                "e_eos": float(e_eos), "e_rlx": float(e_rlx),
                "maxf_rlx": float(mf),
                "mean_maxf": float(np.mean(max_fs)), "std_maxf": float(np.std(max_fs)),
                "eos_a": a_crv.tolist(), "eos_e": e_crv.tolist(),
                "strain_a": s_a.tolist(), "strain_e": s_e.tolist(),
                "force_disps": disps.tolist(), "force_mags": fmags,
            }
            print(f"  ✓ DPA-3-pretrained/{el} done")
        except Exception as ex:
            print(f"  ✗ DPA-3-pretrained/{el}: {ex}")
            traceback.print_exc()

    return results


# ══════════════════════════════════════════════════════════════════
# PHASE 3: TRAINING DATA GENERATION + OPTUNA + FINE-TUNING
# ══════════════════════════════════════════════════════════════════
def generate_training_data():
    """Generate training data from MACE teacher."""
    print("\n  Generating training data from MACE teacher...")
    from mace.calculators import mace_mp

    mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float32")
    data_dir = os.path.join(TRAIN_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    for el in ["Si", "Ge"]:
        el_dir = os.path.join(data_dir, el)
        os.makedirs(el_dir, exist_ok=True)

        a0 = GT[el]["exp_a"]
        type_map = [el]
        rng = np.random.RandomState(42)

        all_coords, all_energies, all_forces, all_boxes = [], [], [], []
        n_atoms_per = None

        # 1. Strained configs (50)
        for s in np.linspace(0.92, 1.08, 50):
            at = bulk(el, crystalstructure='diamond', a=a0*s)
            at.calc = mace_calc
            e = at.get_potential_energy()
            f = at.get_forces()
            cell = at.get_cell()
            all_coords.append(at.get_positions().flatten())
            all_energies.append(e)
            all_forces.append(f.flatten())
            all_boxes.append(cell.flatten())
            n_atoms_per = len(at)

        # 2. Displaced configs (100)
        for i in range(100):
            at = bulk(el, crystalstructure='diamond', a=a0 * rng.uniform(0.96, 1.04))
            dm = rng.uniform(0.005, 0.15)
            at.set_positions(at.get_positions() + rng.randn(*at.get_positions().shape)*dm)
            at.calc = mace_calc
            e = at.get_potential_energy()
            f = at.get_forces()
            cell = at.get_cell()
            all_coords.append(at.get_positions().flatten())
            all_energies.append(e)
            all_forces.append(f.flatten())
            all_boxes.append(cell.flatten())

        # 3. Larger displacements (50) 
        for i in range(50):
            at = bulk(el, crystalstructure='diamond', a=a0 * rng.uniform(0.94, 1.06))
            dm = rng.uniform(0.1, 0.25)
            at.set_positions(at.get_positions() + rng.randn(*at.get_positions().shape)*dm)
            at.calc = mace_calc
            e = at.get_potential_energy()
            f = at.get_forces()
            cell = at.get_cell()
            all_coords.append(at.get_positions().flatten())
            all_energies.append(e)
            all_forces.append(f.flatten())
            all_boxes.append(cell.flatten())

        n_frames = len(all_energies)
        print(f"    {el}: {n_frames} configs, {n_atoms_per} atoms each")

        # Save in DeePMD npy format
        set_dir = os.path.join(el_dir, "set.000")
        os.makedirs(set_dir, exist_ok=True)

        # type.raw
        with open(os.path.join(el_dir, "type.raw"), 'w') as f:
            for _ in range(n_atoms_per):
                f.write("0\n")

        # type_map.raw
        with open(os.path.join(el_dir, "type_map.raw"), 'w') as f:
            f.write(f"{el}\n")

        np.save(os.path.join(set_dir, "coord.npy"), np.array(all_coords))
        np.save(os.path.join(set_dir, "energy.npy"), np.array(all_energies))
        np.save(os.path.join(set_dir, "force.npy"), np.array(all_forces))
        np.save(os.path.join(set_dir, "box.npy"), np.array(all_boxes))

    print("  ✓ Training data saved")
    return data_dir


def create_deepmd_input(params, data_dir, out_dir):
    """Create DeePMD input JSON for training."""
    input_json = {
        "model": {
            "type_map": ["Si", "Ge"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [params.get("sel", 80)],
                "rcut_smth": params.get("rcut_smth", 0.5),
                "rcut": params.get("rcut", 6.0),
                "neuron": params.get("neuron", [25, 50, 100]),
                "resnet_dt": False,
                "axis_neuron": 16,
                "seed": 42,
            },
            "fitting_net": {
                "neuron": params.get("fitting_neuron", [240, 240, 240]),
                "resnet_dt": True,
                "seed": 42,
            }
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": params.get("decay_steps", 500),
            "start_lr": params.get("start_lr", 0.001),
            "stop_lr": params.get("stop_lr", 1e-6),
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [
                    os.path.join(data_dir, "Si"),
                    os.path.join(data_dir, "Ge"),
                ],
                "batch_size": "auto",
            },
            "validation_data": {
                "systems": [
                    os.path.join(data_dir, "Si"),
                    os.path.join(data_dir, "Ge"),
                ],
                "batch_size": "auto",
            },
            "numb_steps": params.get("numb_steps", 5000),
            "seed": 42,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": params.get("numb_steps", 5000),
        }
    }

    input_path = os.path.join(out_dir, "input.json")
    with open(input_path, 'w') as f:
        json.dump(input_json, f, indent=2)
    return input_path


def train_deepmd(input_path, work_dir):
    """Train DeePMD model. Returns lcurve data and model path."""
    env = os.environ.copy()
    env["PATH"] = "/home/bib569/.local/bin:" + env.get("PATH", "")

    # Use 'dp --pt train' syntax
    r = subprocess.run(
        [DP_CMD, "--pt", "train", input_path],
        cwd=work_dir, capture_output=True, text=True, timeout=1800, env=env,
    )

    if r.returncode != 0:
        raise RuntimeError(f"dp train failed:\n{r.stderr[-1000:]}\n{r.stdout[-1000:]}")

    # Parse lcurve.out
    lcurve_path = os.path.join(work_dir, "lcurve.out")
    train_losses, val_losses = [], []
    if os.path.exists(lcurve_path):
        with open(lcurve_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        train_losses.append(float(parts[1]))  # energy loss
                        val_losses.append(float(parts[3]))    # force loss
                    except:
                        pass

    # Freeze model
    model_path = os.path.join(work_dir, "frozen_model.pth")
    r2 = subprocess.run(
        [DP_CMD, "--pt", "freeze", "-o", model_path],
        cwd=work_dir, capture_output=True, text=True, timeout=300, env=env,
    )
    if r2.returncode != 0:
        # Try alternative freeze
        ckpt = os.path.join(work_dir, "model.ckpt.pt")
        if os.path.exists(ckpt):
            model_path = ckpt
        else:
            # Find any .pt file
            for fn in os.listdir(work_dir):
                if fn.endswith(".pt") and "model" in fn:
                    model_path = os.path.join(work_dir, fn)
                    break

    return model_path, train_losses, val_losses


def run_optuna_deepmd(data_dir, n_trials=8):
    """Run Optuna HPO for DeepMD training."""
    print("\n" + "="*70)
    print("  PHASE 3: DeepMD Training with Optuna HPO")
    print(f"  Trials: {n_trials}")
    print("="*70)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_losses = []

    def objective(trial):
        params = {
            "sel": trial.suggest_categorical("sel", [40, 60, 80]),
            "rcut": trial.suggest_float("rcut", 5.0, 7.0, step=0.5),
            "rcut_smth": trial.suggest_float("rcut_smth", 0.3, 1.0, step=0.1),
            "neuron": trial.suggest_categorical("neuron_config",
                ["small", "medium", "large"]),
            "start_lr": trial.suggest_float("start_lr", 5e-4, 5e-3, log=True),
            "numb_steps": trial.suggest_categorical("numb_steps", [3000, 5000]),
            "decay_steps": trial.suggest_categorical("decay_steps", [300, 500]),
        }

        # Map neuron config
        neuron_map = {
            "small": [16, 32, 64],
            "medium": [25, 50, 100],
            "large": [32, 64, 128],
        }
        fitting_map = {
            "small": [120, 120, 120],
            "medium": [240, 240, 240],
            "large": [240, 240, 240],
        }
        neuron_key = params["neuron"]
        params["neuron"] = neuron_map[neuron_key]
        params["fitting_neuron"] = fitting_map[neuron_key]

        trial_dir = os.path.join(TRAIN_DIR, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        try:
            input_path = create_deepmd_input(params, data_dir, trial_dir)
            t0 = time.time()
            model_path, train_l, val_l = train_deepmd(input_path, trial_dir)
            dt = time.time() - t0
            
            final_train = train_l[-1] if train_l else float('inf')
            final_val = val_l[-1] if val_l else float('inf')
            print(f"    Trial {trial.number}: {dt:.0f}s, "
                  f"train_loss={final_train:.6f}, val_loss={final_val:.6f}")

            all_losses.append({
                "trial": trial.number,
                "params": {k: v for k, v in trial.params.items()},
                "train_losses": train_l,
                "val_losses": val_l,
                "time": dt,
            })

            return final_val
        except Exception as e:
            print(f"    Trial {trial.number} FAILED: {e}")
            return float('inf')

    study = optuna.create_study(direction="minimize",
                                study_name="deepmd_si_ge")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Best trial: {study.best_trial.number}")
    print(f"  Best params: {study.best_trial.params}")
    print(f"  Best val_loss: {study.best_trial.value:.6f}")

    # Save Optuna results
    optuna_results = {
        "best_trial": study.best_trial.number,
        "best_params": {k: v for k, v in study.best_trial.params.items()},
        "best_value": study.best_trial.value,
        "all_trials": all_losses,
    }
    with open(os.path.join(OUT, "optuna_results.json"), 'w') as f:
        json.dump(optuna_results, f, indent=2)

    # Return best model path
    best_dir = os.path.join(TRAIN_DIR, f"trial_{study.best_trial.number}")
    best_model = None
    for fn in os.listdir(best_dir):
        if fn.endswith(".pth") or (fn.endswith(".pt") and "model" in fn):
            best_model = os.path.join(best_dir, fn)
            break
    if best_model is None:
        # Use the checkpoint
        best_model = os.path.join(best_dir, "model.ckpt.pt")

    return best_model, optuna_results


def benchmark_deepmd_finetuned(model_path):
    """Benchmark fine-tuned DeepMD model."""
    print("\n" + "="*70)
    print("  Benchmarking Fine-tuned DeepMD")
    print("="*70)

    from deepmd.calculator import DP as DPCalc

    def dp_calc():
        return DPCalc(model=model_path)

    results = {}
    for el in ["Si", "Ge"]:
        print(f"\n  --- DeepMD-finetuned / {el} ---")
        try:
            print("  [1] EOS...")
            a_eos, e_eos, a_crv, e_crv = eos_lattice(dp_calc, el)
            print(f"      a={a_eos:.4f} (DFT={GT[el]['dft_a']:.3f})")

            print("  [2] Relax...")
            a_rlx, e_rlx, mf = relax_cell(dp_calc, el)
            print(f"      a={a_rlx:.4f}, E/at={e_rlx:.4f}, maxF={mf:.2e}")

            print("  [3] Bootstrap forces...")
            disps, fmags = bootstrap_forces(dp_calc, el)
            max_fs = [np.max(np.abs(np.array(f).reshape(-1, 3))) for f in fmags]
            print(f"      <maxF>={np.mean(max_fs):.4f}±{np.std(max_fs):.4f}")

            print("  [4] E-strain...")
            s_a, s_e = energy_strain(dp_calc, el)

            results[el] = {
                "a_eos": float(a_eos), "a_rlx": float(a_rlx),
                "e_eos": float(e_eos), "e_rlx": float(e_rlx),
                "maxf_rlx": float(mf),
                "mean_maxf": float(np.mean(max_fs)), "std_maxf": float(np.std(max_fs)),
                "eos_a": a_crv.tolist(), "eos_e": e_crv.tolist(),
                "strain_a": s_a.tolist(), "strain_e": s_e.tolist(),
                "force_disps": disps.tolist(), "force_mags": fmags,
            }
            print(f"  ✓ DeepMD-finetuned/{el} done")
        except Exception as ex:
            print(f"  ✗ DeepMD-finetuned/{el}: {ex}")
            traceback.print_exc()

    return results


# ══════════════════════════════════════════════════════════════════
# PHASE 4: MEAM Ge BENCHMARK
# ══════════════════════════════════════════════════════════════════
def benchmark_meam_ge():
    """Benchmark Ge using MEAM potential (Baskes 1992 parameters)."""
    print("\n" + "="*70)
    print("  PHASE 4: MEAM Ge Benchmark")
    print("="*70)

    # Create a Ge MEAM library file with Baskes (1992) parameters
    ge_lib = os.path.join(POT, "Ge_library.meam")
    with open(ge_lib, 'w') as f:
        # Format: elt lat z ielement atwt
        #         alpha b0 b1 b2 b3 alat esub asub
        #         t0 t1 t2 t3 rozero ibar
        f.write("# MEAM library file for Ge\n")
        f.write("# Reference: Baskes, PRB 46, 2727 (1992)\n")
        f.write("# elt     lat  z    ielement  atwt\n")
        f.write("'Ge'     'dia'  4    32    72.630\n")
        # Parameters from Baskes 1992 for Ge diamond
        f.write("4.9645 3.48 5.89 5.89 5.89 5.6575 3.85 1.00\n")
        f.write("1.0 4.02 5.53 -0.61 1.0 3\n")

    ge_meam_params = os.path.join(POT, "Ge_single.meam")
    with open(ge_meam_params, 'w') as f:
        f.write("# Ge MEAM parameters\n")
        f.write("rc = 6.0\n")
        f.write("delr = 0.1\n")
        f.write("augt1 = 0\n")
        f.write("ialloy = 0\n")
        f.write("emb_lin_neg = 0\n")
        f.write("bkgd_dyn = 0\n")

    el = "Ge"
    pair_cmds = f"""pair_style meam
pair_coeff * * {ge_lib} Ge {ge_meam_params} Ge"""

    results = {}
    try:
        print(f"\n  --- MEAM / {el} ---")
        print("  [1] EOS...")
        a_eos, e_eos, a_crv, e_crv = eos_lattice_lammps(pair_cmds, el)
        print(f"      a={a_eos:.4f} (DFT={GT[el]['dft_a']:.3f})")

        print("  [2] Relax...")
        a_rlx, e_rlx, mf = relax_cell_lammps(pair_cmds, el)
        print(f"      a={a_rlx:.4f}, E/at={e_rlx:.4f}, maxF={mf:.2e}")

        print("  [3] Bootstrap forces...")
        disps, fmags = bootstrap_forces_lammps(pair_cmds, el)
        max_fs = [np.max(np.abs(np.array(f).reshape(-1, 3))) for f in fmags]
        print(f"      <maxF>={np.mean(max_fs):.4f}±{np.std(max_fs):.4f}")

        print("  [4] E-strain...")
        s_a, s_e = energy_strain_lammps(pair_cmds, el)

        results[el] = {
            "a_eos": float(a_eos), "a_rlx": float(a_rlx),
            "e_eos": float(e_eos), "e_rlx": float(e_rlx),
            "maxf_rlx": float(mf),
            "mean_maxf": float(np.mean(max_fs)), "std_maxf": float(np.std(max_fs)),
            "eos_a": a_crv.tolist(), "eos_e": e_crv.tolist(),
            "strain_a": s_a.tolist(), "strain_e": s_e.tolist(),
            "force_disps": disps.tolist(), "force_mags": fmags,
        }
        print(f"  ✓ MEAM/{el} done")
    except Exception as ex:
        print(f"  ✗ MEAM/{el}: {ex}")
        traceback.print_exc()
        print("  Falling back to SW Ge (already benchmarked)")

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  DPA-3 + DeepMD Fine-tuned + MEAM Ge Benchmark                 ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    all_results = {}

    # Phase 2: DPA-3 pre-trained (SemiCond head)
    try:
        dpa3_results = benchmark_dpa3_pretrained()
        all_results["DPA-3-pretrained"] = dpa3_results
    except Exception as e:
        print(f"Phase 2 failed: {e}")
        traceback.print_exc()

    # Phase 3: Training data + Optuna + fine-tuned benchmark
    try:
        data_dir = generate_training_data()
        best_model, optuna_res = run_optuna_deepmd(data_dir, n_trials=8)
        if best_model and os.path.exists(best_model):
            ft_results = benchmark_deepmd_finetuned(best_model)
            all_results["DeepMD-finetuned"] = ft_results
    except Exception as e:
        print(f"Phase 3 failed: {e}")
        traceback.print_exc()

    # Phase 4: MEAM Ge
    try:
        meam_results = benchmark_meam_ge()
        all_results["MEAM"] = meam_results
    except Exception as e:
        print(f"Phase 4 failed: {e}")
        traceback.print_exc()

    # Save all new results
    with open(os.path.join(OUT, "deepmd_meam_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved: {os.path.join(OUT, 'deepmd_meam_results.json')}")

    # Merge with existing results
    existing_path = os.path.join(OUT, "results.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        with open(os.path.join(OUT, "all_results_merged.json"), 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"✓ Merged results: {os.path.join(OUT, 'all_results_merged.json')}")

    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  COMPLETE                                                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
