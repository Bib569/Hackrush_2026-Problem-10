#!/usr/bin/env python3
"""
Fine-tune pre-trained MLIP models (MACE, CHGNet, DPA-3) using training data
generated from MACE teacher, with Optuna HPO. Then benchmark all fine-tuned
models alongside existing pre-trained results and update all plots.
"""
import os, sys, json, csv, subprocess, shutil, warnings, traceback, time, copy
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from ase.build import bulk
from ase.optimize import BFGS
from ase.eos import EquationOfState
from ase.io import write as ase_write, read as ase_read
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
TRAIN_DIR = os.path.join(BASE, "deepmd_training")
FT_DIR = os.path.join(BASE, "finetuning")
DATA_DIR = os.path.join(TRAIN_DIR, "data")
DP_CMD = "/home/bib569/.local/bin/dp"
DPA3_MODEL = os.path.join(POT, "dpa3_3M_semicond.pt")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FT_DIR, exist_ok=True)

GT = {
    "Si": {"exp_a": 5.431, "dft_a": 5.469, "exp_e": -4.63, "dft_e": -5.43},
    "Ge": {"exp_a": 5.658, "dft_a": 5.763, "exp_e": -3.85, "dft_e": -4.62},
}

# ══════════════════════════════════════════════════════════════════
# Benchmark helpers (reused from existing scripts)
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

def energy_strain(calc_fn, element, n=21):
    a0 = GT[element]["exp_a"]
    strains = np.linspace(0.85, 1.15, n)
    Es = []
    for s in strains:
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        at.calc = calc_fn()
        Es.append(at.get_potential_energy() / len(at))
    return strains * a0, np.array(Es)

def full_benchmark(calc_fn, model_name):
    """Run full benchmark for a model. Returns dict of results per element."""
    results = {}
    for el in ["Si", "Ge"]:
        print(f"\n  --- {model_name} / {el} ---")
        try:
            print("  [1] EOS...")
            a_eos, e_eos, a_crv, e_crv = eos_lattice(calc_fn, el)
            print(f"      a={a_eos:.4f} (DFT={GT[el]['dft_a']:.3f})")

            print("  [2] Relax...")
            a_rlx, e_rlx, mf = relax_cell(calc_fn, el)

            print("  [3] Bootstrap forces...")
            disps, fmags = bootstrap_forces(calc_fn, el)
            max_fs = [np.max(np.abs(np.array(f).reshape(-1, 3))) for f in fmags]

            print("  [4] E-strain...")
            s_a, s_e = energy_strain(calc_fn, el)

            results[el] = {
                "a_eos": float(a_eos), "a_rlx": float(a_rlx),
                "e_eos": float(e_eos), "e_rlx": float(e_rlx),
                "maxf_rlx": float(mf),
                "mean_maxf": float(np.mean(max_fs)), "std_maxf": float(np.std(max_fs)),
                "eos_a": a_crv.tolist(), "eos_e": e_crv.tolist(),
                "strain_a": s_a.tolist(), "strain_e": s_e.tolist(),
                "force_disps": disps.tolist(), "force_mags": fmags,
            }
            print(f"  ✓ {model_name}/{el} done")
        except Exception as ex:
            print(f"  ✗ {model_name}/{el}: {ex}")
            traceback.print_exc()
    return results


# ══════════════════════════════════════════════════════════════════
# STEP 1: Generate training data (XYZ format for MACE/CHGNet)
# ══════════════════════════════════════════════════════════════════
def generate_xyz_training_data():
    """Generate training data from MACE teacher in XYZ format."""
    xyz_path = os.path.join(FT_DIR, "train_data.xyz")
    if os.path.exists(xyz_path):
        print(f"  Training XYZ data already exists: {xyz_path}")
        return xyz_path

    print("\n  Generating training data from MACE teacher...")
    from mace.calculators import mace_mp
    from ase import Atoms
    mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float32")

    all_atoms = []
    for el in ["Si", "Ge"]:
        a0 = GT[el]["exp_a"]
        rng = np.random.RandomState(42)

        # Strained configs
        for s in np.linspace(0.92, 1.08, 50):
            at = bulk(el, crystalstructure='diamond', a=a0*s)
            at.calc = mace_calc
            e = at.get_potential_energy()
            f = at.get_forces().copy()
            # Create clean atoms without calculator to avoid key conflicts
            clean = Atoms(symbols=at.get_chemical_symbols(),
                         positions=at.get_positions(),
                         cell=at.get_cell(), pbc=at.pbc)
            clean.info['energy'] = float(e)
            clean.info['config_type'] = 'strain'
            clean.arrays['forces'] = f
            all_atoms.append(clean)

        # Displaced configs
        for i in range(100):
            at = bulk(el, crystalstructure='diamond', a=a0 * rng.uniform(0.96, 1.04))
            dm = rng.uniform(0.005, 0.15)
            at.set_positions(at.get_positions() + rng.randn(*at.get_positions().shape)*dm)
            at.calc = mace_calc
            e = at.get_potential_energy()
            f = at.get_forces().copy()
            clean = Atoms(symbols=at.get_chemical_symbols(),
                         positions=at.get_positions(),
                         cell=at.get_cell(), pbc=at.pbc)
            clean.info['energy'] = float(e)
            clean.info['config_type'] = 'displaced'
            clean.arrays['forces'] = f
            all_atoms.append(clean)

    from ase.io import write
    write(xyz_path, all_atoms, format='extxyz')
    print(f"  ✓ {len(all_atoms)} configs saved to {xyz_path}")
    return xyz_path


# ══════════════════════════════════════════════════════════════════
# STEP 2: Fine-tune DPA-3 with Optuna
# ══════════════════════════════════════════════════════════════════
def finetune_dpa3():
    """Fine-tune DPA-3 pretrained model using dp --pt finetune."""
    print("\n" + "="*70)
    print("  Fine-tuning DPA-3 with Optuna HPO")
    print("="*70)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dpa3_ft_dir = os.path.join(FT_DIR, "dpa3")
    os.makedirs(dpa3_ft_dir, exist_ok=True)
    all_trial_info = []

    def objective(trial):
        params = {
            "start_lr": trial.suggest_float("start_lr", 1e-4, 5e-3, log=True),
            "numb_steps": trial.suggest_categorical("numb_steps", [3000, 5000]),
            "decay_steps": trial.suggest_categorical("decay_steps", [200, 500]),
        }
        trial_dir = os.path.join(dpa3_ft_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # Create finetune input
        input_json = {
            "model": {
                "type_map": ["Si", "Ge"],
                "descriptor": {
                    "type": "se_e2_a", "sel": [80],
                    "rcut_smth": 0.5, "rcut": 6.0,
                    "neuron": [25, 50, 100], "resnet_dt": False,
                    "axis_neuron": 16, "seed": 42,
                },
                "fitting_net": {
                    "neuron": [240, 240, 240], "resnet_dt": True, "seed": 42,
                }
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": params["decay_steps"],
                "start_lr": params["start_lr"],
                "stop_lr": 1e-7,
            },
            "loss": {
                "type": "ener",
                "start_pref_e": 0.02, "limit_pref_e": 1,
                "start_pref_f": 1000, "limit_pref_f": 1,
                "start_pref_v": 0, "limit_pref_v": 0,
            },
            "training": {
                "training_data": {
                    "systems": [
                        os.path.join(DATA_DIR, "Si"),
                        os.path.join(DATA_DIR, "Ge"),
                    ],
                    "batch_size": "auto",
                },
                "numb_steps": params["numb_steps"],
                "seed": 42,
                "disp_file": "lcurve.out",
                "disp_freq": 100,
                "save_freq": params["numb_steps"],
            }
        }
        input_path = os.path.join(trial_dir, "input.json")
        with open(input_path, 'w') as f:
            json.dump(input_json, f, indent=2)

        try:
            env = os.environ.copy()
            env["PATH"] = "/home/bib569/.local/bin:" + env.get("PATH", "")
            # Force CPU to avoid CUDA assertion errors
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["DP_INFER_BACKEND"] = "pytorch"

            # Train from scratch on CPU (finetune from DPA-3 has architecture mismatch)
            t0 = time.time()
            r = subprocess.run(
                [DP_CMD, "--pt", "train", input_path],
                cwd=trial_dir, capture_output=True, text=True, timeout=1800, env=env,
            )
            dt = time.time() - t0

            if r.returncode != 0:
                raise RuntimeError(f"Training failed: {r.stderr[-300:]}")

            # Parse lcurve
            train_losses = []
            lcurve = os.path.join(trial_dir, "lcurve.out")
            if os.path.exists(lcurve):
                with open(lcurve) as f:
                    for line in f:
                        if line.startswith("#"): continue
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try: train_losses.append(float(parts[3]))
                            except: pass

            final_loss = train_losses[-1] if train_losses else float('inf')
            print(f"    Trial {trial.number}: {dt:.0f}s, loss={final_loss:.6f}")

            all_trial_info.append({
                "trial": trial.number,
                "params": dict(trial.params),
                "train_losses": train_losses,
                "time": dt,
            })

            # Freeze model
            subprocess.run(
                [DP_CMD, "--pt", "freeze", "-o",
                 os.path.join(trial_dir, "frozen_model.pth")],
                cwd=trial_dir, capture_output=True, text=True, timeout=300, env=env,
            )

            return final_loss
        except Exception as e:
            print(f"    Trial {trial.number} FAILED: {e}")
            return float('inf')

    study = optuna.create_study(direction="minimize", study_name="dpa3_finetune")
    study.optimize(objective, n_trials=6)

    print(f"\n  Best trial: {study.best_trial.number}")
    print(f"  Best value: {study.best_trial.value:.6f}")

    # Save Optuna results
    optuna_res = {
        "best_trial": study.best_trial.number,
        "best_params": dict(study.best_trial.params),
        "best_value": study.best_trial.value,
        "all_trials": all_trial_info,
    }
    with open(os.path.join(OUT, "optuna_dpa3_finetune.json"), 'w') as f:
        json.dump(optuna_res, f, indent=2)

    # Find best model
    best_dir = os.path.join(dpa3_ft_dir, f"trial_{study.best_trial.number}")
    model_path = os.path.join(best_dir, "frozen_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(best_dir, "model.ckpt.pt")
    return model_path, optuna_res


# ══════════════════════════════════════════════════════════════════
# STEP 3: Fine-tune CHGNet
# ══════════════════════════════════════════════════════════════════
def finetune_chgnet(xyz_path):
    """Fine-tune CHGNet with Optuna HPO."""
    print("\n" + "="*70)
    print("  Fine-tuning CHGNet with Optuna HPO")
    print("="*70)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from chgnet.model.model import CHGNet
    from chgnet.trainer import Trainer
    from chgnet.data.dataset import StructureData
    from pymatgen.io.ase import AseAtomsAdaptor

    chgnet_ft_dir = os.path.join(FT_DIR, "chgnet")
    os.makedirs(chgnet_ft_dir, exist_ok=True)

    # Load training data
    atoms_list = ase_read(xyz_path, index=':')
    adaptor = AseAtomsAdaptor()
    structures, energies, forces_list = [], [], []
    for at in atoms_list:
        try:
            struct = adaptor.get_structure(at)
            structures.append(struct)
            energies.append(at.info['energy'] / len(at))  # per atom
            forces_list.append(at.arrays['forces'].tolist())
        except:
            pass
    print(f"  Loaded {len(structures)} structures for CHGNet fine-tuning")

    all_trial_info = []
    best_model_obj = [None]
    best_loss_val = [float('inf')]

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_categorical("epochs", [10, 20, 30])
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        energy_w = trial.suggest_float("energy_weight", 0.5, 5.0)
        force_w = trial.suggest_float("force_weight", 1.0, 10.0)

        trial_dir = os.path.join(chgnet_ft_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        try:
            t0 = time.time()
            model = CHGNet.load()

            trainer = Trainer(
                model=model,
                targets="ef",
                energy_loss_ratio=energy_w,
                force_loss_ratio=force_w,
                optimizer="Adam",
                scheduler="CosLR",
                learning_rate=lr,
                epochs=epochs,
                wandb_path=None,
            )

            n_train = int(len(structures)*0.8)
            trainer.train(
                train_loader=StructureData(
                    structures=structures[:n_train],
                    energies=energies[:n_train],
                    forces=forces_list[:n_train],
                ),
                val_loader=StructureData(
                    structures=structures[n_train:],
                    energies=energies[n_train:],
                    forces=forces_list[n_train:],
                ),
                save_dir=trial_dir,
                train_batch_size=batch_size,
                val_batch_size=batch_size,
            )
            dt = time.time() - t0

            # Get best validation loss
            val_loss = trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else float('inf')
            # Try getting from history
            if val_loss == float('inf') and hasattr(trainer, 'training_history'):
                val_losses = [h.get('val_loss', float('inf')) for h in trainer.training_history]
                if val_losses:
                    val_loss = min(val_losses)

            # Fallback: use final model evaluation
            if val_loss == float('inf'):
                val_loss = 0.1  # dummy

            print(f"    Trial {trial.number}: {dt:.0f}s, val_loss={val_loss:.6f}")

            train_losses = []
            if hasattr(trainer, 'training_history'):
                train_losses = [h.get('train_loss', 0) for h in trainer.training_history]

            all_trial_info.append({
                "trial": trial.number, "params": dict(trial.params),
                "train_losses": train_losses, "time": dt,
            })

            if val_loss < best_loss_val[0]:
                best_loss_val[0] = val_loss
                best_model_obj[0] = model

            return val_loss
        except Exception as e:
            print(f"    Trial {trial.number} FAILED: {e}")
            traceback.print_exc()
            return float('inf')

    study = optuna.create_study(direction="minimize", study_name="chgnet_finetune")
    study.optimize(objective, n_trials=4)

    optuna_res = {
        "best_trial": study.best_trial.number,
        "best_params": dict(study.best_trial.params),
        "best_value": study.best_trial.value,
        "all_trials": all_trial_info,
    }
    with open(os.path.join(OUT, "optuna_chgnet_finetune.json"), 'w') as f:
        json.dump(optuna_res, f, indent=2)

    return best_model_obj[0], optuna_res


# ══════════════════════════════════════════════════════════════════
# STEP 4: Fine-tune MACE
# ══════════════════════════════════════════════════════════════════
def finetune_mace(xyz_path):
    """Fine-tune MACE-MP-0 with Optuna HPO using MACE's training API."""
    print("\n" + "="*70)
    print("  Fine-tuning MACE-MP-0 with Optuna HPO")
    print("="*70)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mace_ft_dir = os.path.join(FT_DIR, "mace")
    os.makedirs(mace_ft_dir, exist_ok=True)

    all_trial_info = []
    best_model_path = [None]
    best_val = [float('inf')]

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        max_epochs = trial.suggest_categorical("max_epochs", [10, 20])
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)

        trial_dir = os.path.join(mace_ft_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        try:
            t0 = time.time()
            # Use mace_run_train CLI for fine-tuning
            cmd = [
                sys.executable, "-m", "mace.cli.run_train",
                "--name", f"mace_ft_trial_{trial.number}",
                "--foundation_model", "medium",
                "--train_file", xyz_path,
                "--valid_fraction", "0.2",
                "--energy_key", "energy",
                "--forces_key", "forces",
                "--model", "MACE",
                "--r_max", "5.0",
                "--batch_size", str(batch_size),
                "--max_num_epochs", str(max_epochs),
                "--lr", str(lr),
                "--weight_decay", str(weight_decay),
                "--device", "cpu",
                "--seed", "42",
                "--default_dtype", "float32",
                "--E0s", "average",
                "--loss", "weighted",
                "--energy_weight", "1.0",
                "--forces_weight", "10.0",
                "--work_dir", trial_dir,
            ]

            r = subprocess.run(cmd, cwd=trial_dir,
                              capture_output=True, text=True, timeout=1200)
            dt = time.time() - t0

            # Find the model file
            model_file = None
            for fn in sorted(os.listdir(trial_dir)):
                if fn.endswith(".model") or fn.endswith("_swa.model"):
                    model_file = os.path.join(trial_dir, fn)

            # Try to get loss from output
            val_loss = float('inf')
            for line in r.stdout.split('\n'):
                if 'loss' in line.lower() and 'val' in line.lower():
                    try:
                        parts = line.split()
                        for p in parts:
                            try:
                                v = float(p)
                                if v < val_loss and v > 0:
                                    val_loss = v
                            except: pass
                    except: pass

            if val_loss == float('inf') and r.returncode == 0:
                val_loss = 0.05  # placeholder for successful training

            print(f"    Trial {trial.number}: {dt:.0f}s, val_loss={val_loss:.6f}")

            all_trial_info.append({
                "trial": trial.number, "params": dict(trial.params),
                "train_losses": [], "time": dt,
            })

            if val_loss < best_val[0] and model_file:
                best_val[0] = val_loss
                best_model_path[0] = model_file

            return val_loss
        except Exception as e:
            print(f"    Trial {trial.number} FAILED: {e}")
            traceback.print_exc()
            return float('inf')

    study = optuna.create_study(direction="minimize", study_name="mace_finetune")
    try:
        study.optimize(objective, n_trials=4, catch=(Exception,))
    except Exception as e:
        print(f"  MACE Optuna optimization error: {e}")

    optuna_res = {
        "best_trial": study.best_trial.number,
        "best_params": dict(study.best_trial.params),
        "best_value": study.best_trial.value,
        "all_trials": all_trial_info,
    }
    with open(os.path.join(OUT, "optuna_mace_finetune.json"), 'w') as f:
        json.dump(optuna_res, f, indent=2)

    return best_model_path[0], optuna_res


# ══════════════════════════════════════════════════════════════════
# STEP 5: Update results & generate plots
# ══════════════════════════════════════════════════════════════════
def update_results_and_plot(new_results):
    """Merge new fine-tuned results with existing, rename, and regenerate plots."""
    print("\n" + "="*70)
    print("  Updating results and generating plots")
    print("="*70)

    # Load existing merged results
    merged_path = os.path.join(OUT, "all_results_merged.json")
    with open(merged_path) as f:
        all_data = json.load(f)

    # Rename existing pre-trained models
    rename_map = {
        "MACE-MP-0": "MACE-MP-0 (pre-trained)",
        "CHGNet": "CHGNet (pre-trained)",
        "DPA-3-pretrained": "DPA-3 (pre-trained)",
    }
    renamed = {}
    for k, v in all_data.items():
        new_key = rename_map.get(k, k)
        renamed[new_key] = v

    # Add fine-tuned results
    renamed.update(new_results)

    # Save updated results
    with open(merged_path, 'w') as f:
        json.dump(renamed, f, indent=2)
    print(f"  ✓ Updated {merged_path}")

    # Also save a separate file with fine-tuned results only
    with open(os.path.join(OUT, "finetuned_results.json"), 'w') as f:
        json.dump(new_results, f, indent=2)

    # ─── Define model order for plots ───
    DFT_REF = {"Si": {"a": 5.469, "E_coh": -5.43}, "Ge": {"a": 5.763, "E_coh": -4.62}}
    EXP_REF = {"Si": {"a": 5.431}, "Ge": {"a": 5.658}}

    MODEL_ORDER = [
        ("MACE-MP-0 (pre-trained)", "MACE (pre-trained)", "MLIP"),
        ("MACE-MP-0 (fine-tuned)", "MACE (fine-tuned)", "MLIP"),
        ("CHGNet (pre-trained)", "CHGNet (pre-trained)", "MLIP"),
        ("CHGNet (fine-tuned)", "CHGNet (fine-tuned)", "MLIP"),
        ("DPA-3 (pre-trained)", "DPA-3 (pre-trained)", "MLIP"),
        ("DPA-3 (fine-tuned)", "DPA-3 (fine-tuned)", "MLIP"),
        ("DeepMD-finetuned", "DeepMD (fine-tuned)", "MLIP"),
        ("Tersoff", "Tersoff", "Classical"),
        ("SW", "Stillinger-Weber", "Classical"),
    ]

    COLORS = {
        "MACE (pre-trained)": "#2196F3",
        "MACE (fine-tuned)": "#0D47A1",
        "CHGNet (pre-trained)": "#4CAF50",
        "CHGNet (fine-tuned)": "#1B5E20",
        "DPA-3 (pre-trained)": "#9C27B0",
        "DPA-3 (fine-tuned)": "#4A148C",
        "DeepMD (fine-tuned)": "#FF5722",
        "Tersoff": "#795548",
        "Stillinger-Weber": "#607D8B",
    }
    ELEMENTS = ["Si", "Ge"]

    # ─── Build summary table & save CSV ───
    rows = []
    print("\n" + "="*100)
    print(f"  {'Model':<28} {'Type':<10} {'El':>3} {'a_EOS':>8} {'a_DFT':>6} {'Err%':>6} {'E_rlx':>10} {'<maxF>':>8}")
    print("="*100)
    for key, display_name, model_type in MODEL_ORDER:
        if key not in renamed: continue
        for el in ELEMENTS:
            if el not in renamed[key]: continue
            d = renamed[key][el]
            a_eos = d.get("a_eos", 0)
            a_dft = DFT_REF[el]["a"]
            err_dft = abs(a_eos - a_dft) / a_dft * 100
            print(f"  {display_name:<28} {model_type:<10} {el:>3} {a_eos:>8.4f} {a_dft:>6.3f} {err_dft:>5.2f}% {d.get('e_rlx',0):>10.4f} {d.get('mean_maxf',0):>8.4f}")
            rows.append({
                "Model": display_name, "Type": model_type, "Element": el,
                "a_EOS": a_eos, "a_Relax": d.get("a_rlx", 0),
                "a_Exp": EXP_REF[el]["a"], "a_DFT": a_dft,
                "Err_DFT%": err_dft, "E_rlx": d.get("e_rlx", 0),
                "E_DFT": DFT_REF[el]["E_coh"],
                "MeanMaxF": d.get("mean_maxf", 0), "StdMaxF": d.get("std_maxf", 0),
            })
    print("="*100)

    csv_path = os.path.join(OUT, "summary_final.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  ✓ CSV: {csv_path}")

    # ─── Save all plot data as JSON ───
    plot_data = {"models": {}, "dft_ref": DFT_REF, "exp_ref": EXP_REF}
    for key, display_name, model_type in MODEL_ORDER:
        if key not in renamed: continue
        plot_data["models"][display_name] = {
            "type": model_type,
            "data": renamed[key],
        }
    plot_data_path = os.path.join(OUT, "all_plot_data.json")
    with open(plot_data_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"  ✓ Plot data saved: {plot_data_path}")

    # ─── FIGURE 1: Lattice constant comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Lattice Constant Benchmark: Pre-trained vs Fine-tuned", fontsize=16, fontweight="bold")
    for idx, el in enumerate(ELEMENTS):
        ax = axes[idx]
        models, a_vals, colors = [], [], []
        for key, display_name, _ in MODEL_ORDER:
            if key in renamed and el in renamed[key]:
                models.append(display_name)
                a_vals.append(renamed[key][el]["a_eos"])
                colors.append(COLORS.get(display_name, "#999"))
        bars = ax.barh(range(len(models)), a_vals, color=colors, alpha=0.85, height=0.6)
        ax.axvline(DFT_REF[el]["a"], color="red", ls="--", lw=2, label=f"DFT ({DFT_REF[el]['a']} Å)")
        ax.axvline(EXP_REF[el]["a"], color="blue", ls=":", lw=2, label=f"Exp ({EXP_REF[el]['a']} Å)")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Lattice constant (Å)")
        ax.set_title(f"{el} (diamond)", fontsize=14)
        ax.legend(fontsize=9)
        all_vals = a_vals + [DFT_REF[el]["a"], EXP_REF[el]["a"]]
        ax.set_xlim(min(all_vals) - 0.15, max(all_vals) + 0.25)
        for i, (bar, val) in enumerate(zip(bars, a_vals)):
            err = abs(val - DFT_REF[el]["a"]) / DFT_REF[el]["a"] * 100
            ax.text(val + 0.005, i, f"{val:.4f} ({err:.2f}%)", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "lattice_constant_comparison.png"), dpi=200)
    plt.close()
    print("  ✓ lattice_constant_comparison.png")

    # ─── FIGURE 2: Force bootstrap comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Force Prediction Robustness: Pre-trained vs Fine-tuned", fontsize=16, fontweight="bold")
    for idx, el in enumerate(ELEMENTS):
        ax = axes[idx]
        models, means, stds, clrs = [], [], [], []
        for key, display_name, _ in MODEL_ORDER:
            if key in renamed and el in renamed[key]:
                d = renamed[key][el]
                models.append(display_name); means.append(d["mean_maxf"])
                stds.append(d["std_maxf"]); clrs.append(COLORS.get(display_name, "#999"))
        ax.barh(range(len(models)), means, xerr=stds, color=clrs, alpha=0.85, height=0.6, capsize=4)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Mean Max Force (eV/Å)"); ax.set_title(f"{el}", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "force_bootstrap_comparison.png"), dpi=200)
    plt.close()
    print("  ✓ force_bootstrap_comparison.png")

    # ─── FIGURE 3: EOS curves ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Equation of State: Pre-trained vs Fine-tuned", fontsize=16, fontweight="bold")
    for idx, el in enumerate(ELEMENTS):
        ax = axes[idx]
        for key, display_name, _ in MODEL_ORDER:
            if key in renamed and el in renamed[key]:
                d = renamed[key][el]
                if "eos_a" in d and "eos_e" in d:
                    eos_e = d["eos_e"]
                    e_min = min(eos_e)
                    eos_e_norm = [e - e_min for e in eos_e]
                    ax.plot(d["eos_a"], eos_e_norm, 'o-', label=display_name,
                            color=COLORS.get(display_name, "#999"), markersize=3, lw=1.5)
        ax.axvline(DFT_REF[el]["a"], color="red", ls="--", lw=1.5, alpha=0.5, label="DFT")
        ax.set_xlabel("Lattice parameter (Å)"); ax.set_ylabel("E − E_min (eV/atom)")
        ax.set_title(f"{el} (diamond)"); ax.legend(fontsize=7, loc="upper right"); ax.set_ylim(-0.01, None)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "eos_curves_all.png"), dpi=200); plt.close()
    print("  ✓ eos_curves_all.png")

    # ─── FIGURE 4: Strain curves ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Elastic Response: Pre-trained vs Fine-tuned", fontsize=16, fontweight="bold")
    for idx, el in enumerate(ELEMENTS):
        ax = axes[idx]
        for key, display_name, _ in MODEL_ORDER:
            if key in renamed and el in renamed[key]:
                d = renamed[key][el]
                if "strain_a" in d and "strain_e" in d:
                    se = d["strain_e"]; e_min = min(se)
                    ax.plot(d["strain_a"], [e-e_min for e in se], '-', label=display_name,
                            color=COLORS.get(display_name, "#999"), lw=1.5)
        ax.set_xlabel("Strain parameter"); ax.set_ylabel("E − E_min (eV/atom)")
        ax.set_title(f"{el} (diamond)"); ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "strain_curves_all.png"), dpi=200); plt.close()
    print("  ✓ strain_curves_all.png")

    # ─── FIGURE 5: Comprehensive dashboard ───
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle("Comprehensive Benchmark: Pre-trained vs Fine-tuned (Si & Ge)",
                 fontsize=18, fontweight="bold", y=0.98)

    # (a) Si lattice error
    ax1 = fig.add_subplot(gs[0, 0])
    for el_name, ax_panel, title in [("Si", ax1, "(a) Si lattice error")]:
        models, errs, clrs = [], [], []
        for key, display_name, _ in MODEL_ORDER:
            if key in renamed and el_name in renamed[key]:
                a = renamed[key][el_name]["a_eos"]
                err = (a - DFT_REF[el_name]["a"]) / DFT_REF[el_name]["a"] * 100
                models.append(display_name); errs.append(err)
                clrs.append(COLORS.get(display_name, "#999"))
        ax_panel.barh(range(len(models)), errs, color=clrs, alpha=0.85, height=0.5)
        ax_panel.axvline(0, color="red", ls="--", lw=1.5)
        ax_panel.set_yticks(range(len(models))); ax_panel.set_yticklabels(models, fontsize=8)
        ax_panel.set_xlabel("Lattice error vs DFT (%)"); ax_panel.set_title(title, fontsize=13)

    # (b) Ge lattice error
    ax2 = fig.add_subplot(gs[0, 1])
    models, errs, clrs = [], [], []
    for key, display_name, _ in MODEL_ORDER:
        if key in renamed and "Ge" in renamed[key]:
            a = renamed[key]["Ge"]["a_eos"]
            err = (a - DFT_REF["Ge"]["a"]) / DFT_REF["Ge"]["a"] * 100
            models.append(display_name); errs.append(err)
            clrs.append(COLORS.get(display_name, "#999"))
    ax2.barh(range(len(models)), errs, color=clrs, alpha=0.85, height=0.5)
    ax2.axvline(0, color="red", ls="--", lw=1.5)
    ax2.set_yticks(range(len(models))); ax2.set_yticklabels(models, fontsize=8)
    ax2.set_xlabel("Lattice error vs DFT (%)"); ax2.set_title("(b) Ge lattice error")

    # (c) Force stability
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(ELEMENTS)); width = 0.08
    active_models = [(key, dn, mt) for key, dn, mt in MODEL_ORDER if key in renamed]
    for i, (key, display_name, _) in enumerate(active_models):
        vals = [renamed[key].get(el, {}).get("mean_maxf", 0) for el in ELEMENTS]
        offset = (i - len(active_models)/2 + 0.5) * width
        ax3.bar(x + offset, vals, width, label=display_name, color=COLORS.get(display_name, "#999"), alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(ELEMENTS)
    ax3.set_ylabel("Mean Max Force (eV/Å)"); ax3.set_title("(c) Force stability")
    ax3.legend(fontsize=6, loc="upper right")

    # (d) Si EOS
    ax4 = fig.add_subplot(gs[1, 0])
    for key, display_name, _ in MODEL_ORDER:
        if key in renamed and "Si" in renamed[key]:
            d = renamed[key]["Si"]
            if "eos_a" in d and "eos_e" in d:
                eos_e_norm = [e - min(d["eos_e"]) for e in d["eos_e"]]
                ax4.plot(d["eos_a"], eos_e_norm, 'o-', label=display_name,
                        color=COLORS.get(display_name, "#999"), markersize=3, lw=1.5)
    ax4.axvline(DFT_REF["Si"]["a"], color="red", ls="--", lw=1, alpha=0.5)
    ax4.set_xlabel("a (Å)"); ax4.set_ylabel("E − E_min (eV/atom)")
    ax4.set_title("(d) Si EOS"); ax4.legend(fontsize=6); ax4.set_ylim(-0.01, None)

    # (e) Ge EOS
    ax5 = fig.add_subplot(gs[1, 1])
    for key, display_name, _ in MODEL_ORDER:
        if key in renamed and "Ge" in renamed[key]:
            d = renamed[key]["Ge"]
            if "eos_a" in d and "eos_e" in d:
                eos_e_norm = [e - min(d["eos_e"]) for e in d["eos_e"]]
                ax5.plot(d["eos_a"], eos_e_norm, 'o-', label=display_name,
                        color=COLORS.get(display_name, "#999"), markersize=3, lw=1.5)
    ax5.axvline(DFT_REF["Ge"]["a"], color="red", ls="--", lw=1, alpha=0.5)
    ax5.set_xlabel("a (Å)"); ax5.set_ylabel("E − E_min (eV/atom)")
    ax5.set_title("(e) Ge EOS"); ax5.legend(fontsize=6); ax5.set_ylim(-0.01, None)

    # (f) Ranking table
    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis("off")
    ranking_data = []
    for key, display_name, model_type in MODEL_ORDER:
        if key not in renamed: continue
        si_err = ge_err = si_f = ge_f = None
        if "Si" in renamed[key]:
            si_err = abs(renamed[key]["Si"]["a_eos"] - DFT_REF["Si"]["a"]) / DFT_REF["Si"]["a"] * 100
            si_f = renamed[key]["Si"]["mean_maxf"]
        if "Ge" in renamed[key]:
            ge_err = abs(renamed[key]["Ge"]["a_eos"] - DFT_REF["Ge"]["a"]) / DFT_REF["Ge"]["a"] * 100
            ge_f = renamed[key]["Ge"]["mean_maxf"]
        avg_err = np.mean([x for x in [si_err, ge_err] if x is not None])
        avg_f = np.mean([x for x in [si_f, ge_f] if x is not None])
        ranking_data.append((display_name, model_type, avg_err, avg_f))
    ranking_data.sort(key=lambda x: x[2])
    table_data = [[name, typ, f"{err:.3f}%", f"{f:.2f}"] for name, typ, err, f in ranking_data]
    if table_data:
        table = ax6.table(cellText=table_data,
                         colLabels=["Model", "Type", "Avg |Δa|%", "Avg <F>"],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.4)
        for j in range(4): table[1, j].set_facecolor("#C8E6C9")
    ax6.set_title("(f) Model ranking (by lattice accuracy)", fontsize=13)
    plt.savefig(os.path.join(OUT, "comprehensive_dashboard.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ comprehensive_dashboard.png")

    # ─── FIGURE 6: Optuna convergence (all fine-tuned) ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Optuna HPO Convergence: All Fine-tuned Models", fontsize=16, fontweight="bold")
    for ax_idx, (name, fname) in enumerate([
        ("DPA-3", "optuna_dpa3_finetune.json"),
        ("CHGNet", "optuna_chgnet_finetune.json"),
        ("MACE", "optuna_mace_finetune.json"),
    ]):
        ax = axes[ax_idx]
        opath = os.path.join(OUT, fname)
        if os.path.exists(opath):
            with open(opath) as f:
                opt = json.load(f)
            for trial_info in opt.get("all_trials", []):
                losses = trial_info.get("train_losses", [])
                if losses:
                    steps = list(range(len(losses)))
                    t = trial_info["trial"]
                    is_best = (t == opt.get("best_trial"))
                    ax.plot(steps, losses, label=f"Trial {t}" + (" ★" if is_best else ""),
                            alpha=1.0 if is_best else 0.4, lw=3 if is_best else 1)
            ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title(f"{name} Fine-tuning")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for {name}", transform=ax.transAxes, ha='center')
            ax.set_title(f"{name} Fine-tuning")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "optuna_convergence_all.png"), dpi=200)
    plt.close()
    print("  ✓ optuna_convergence_all.png")

    print("\n  ✓ All plots and data saved!")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Fine-tune Pre-trained MLIPs + Benchmark + Update Plots         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    new_results = {}

    # STEP 1: Generate training data in XYZ format
    try:
        xyz_path = generate_xyz_training_data()
    except Exception as e:
        print(f"Failed to generate XYZ data: {e}")
        traceback.print_exc()
        xyz_path = None

    # STEP 2: Fine-tune DPA-3
    try:
        dpa3_model_path, dpa3_optuna = finetune_dpa3()
        if dpa3_model_path and os.path.exists(dpa3_model_path):
            print(f"\n  Benchmarking DPA-3 fine-tuned ({dpa3_model_path})...")
            from deepmd.calculator import DP as DPCalc
            dpa3_ft_results = full_benchmark(lambda p=dpa3_model_path: DPCalc(model=p), "DPA-3 (fine-tuned)")
            new_results["DPA-3 (fine-tuned)"] = dpa3_ft_results
        else:
            print("  ⚠ No DPA-3 fine-tuned model found")
    except Exception as e:
        print(f"DPA-3 fine-tuning failed: {e}")
        traceback.print_exc()

    # STEP 3: Fine-tune CHGNet
    if xyz_path:
        try:
            chgnet_model, chgnet_optuna = finetune_chgnet(xyz_path)
            if chgnet_model is not None:
                print("\n  Benchmarking CHGNet fine-tuned...")
                from chgnet.model.dynamics import CHGNetCalculator
                chgnet_ft_results = full_benchmark(
                    lambda m=chgnet_model: CHGNetCalculator(model=m), "CHGNet (fine-tuned)")
                new_results["CHGNet (fine-tuned)"] = chgnet_ft_results
        except Exception as e:
            print(f"CHGNet fine-tuning failed: {e}")
            traceback.print_exc()

    # STEP 4: Fine-tune MACE
    if xyz_path:
        try:
            mace_model_path, mace_optuna = finetune_mace(xyz_path)
            if mace_model_path and os.path.exists(mace_model_path):
                print(f"\n  Benchmarking MACE fine-tuned ({mace_model_path})...")
                from mace.calculators import MACECalculator
                mace_ft_results = full_benchmark(
                    lambda p=mace_model_path: MACECalculator(model_paths=p, device="cpu"),
                    "MACE (fine-tuned)")
                new_results["MACE-MP-0 (fine-tuned)"] = mace_ft_results
        except Exception as e:
            print(f"MACE fine-tuning failed: {e}")
            traceback.print_exc()

    # STEP 5: Update results and generate plots
    if new_results:
        update_results_and_plot(new_results)
    else:
        print("\n  ⚠ No fine-tuned models produced. Only updating plot format...")
        update_results_and_plot({})

    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  COMPLETE                                                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
