#!/usr/bin/env python3
"""
Phase 3 fixed: DeepMD CPU Training with sel=auto + benchmark fine-tuned model.
Merges with existing DPA-3 + Tersoff results.
"""
import os, sys, json, subprocess, tempfile, shutil, warnings, traceback, time, copy
import numpy as np
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
os.environ['OMP_NUM_THREADS'] = '4'

BASE = "/home/bib569/Workspace/Hackrush_2026-Problem-10"
OUT = os.path.join(BASE, "results_comprehensive")
POT = os.path.join(BASE, "potentials")
DP_CMD = "/home/bib569/.local/bin/dp"
TRAIN_DIR = os.path.join(BASE, "deepmd_training")

GT = {
    "Si": {"exp_a": 5.431, "dft_a": 5.469, "exp_e": -4.63, "dft_e": -5.43},
    "Ge": {"exp_a": 5.658, "dft_a": 5.763, "exp_e": -3.85, "dft_e": -4.62},
}

# ── ASE Benchmark helpers ──
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


# ── Training ──
def create_deepmd_input(params, data_dir, out_dir):
    input_json = {
        "model": {
            "type_map": ["Si", "Ge"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": "auto",  # FIXED: was incorrect list
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
            "start_pref_e": 0.02, "limit_pref_e": 1,
            "start_pref_f": 1000, "limit_pref_f": 1,
            "start_pref_v": 0, "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [os.path.join(data_dir, "Si"), os.path.join(data_dir, "Ge")],
                "batch_size": "auto",
            },
            "numb_steps": params.get("numb_steps", 3000),
            "seed": 42,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": params.get("numb_steps", 3000),
        }
    }
    input_path = os.path.join(out_dir, "input.json")
    with open(input_path, 'w') as f:
        json.dump(input_json, f, indent=2)
    return input_path


def train_deepmd_cpu(input_path, work_dir):
    env = os.environ.copy()
    env["PATH"] = "/home/bib569/.local/bin:" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["DP_DEVICE"] = "cpu"
    
    r = subprocess.run(
        [DP_CMD, "--pt", "train", input_path],
        cwd=work_dir, capture_output=True, text=True, timeout=3600, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"dp train failed:\n{r.stderr[-800:]}\n{r.stdout[-200:]}")

    # Parse lcurve
    lcurve_path = os.path.join(work_dir, "lcurve.out")
    train_losses, val_losses = [], []
    if os.path.exists(lcurve_path):
        with open(lcurve_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        train_losses.append(float(parts[3]))  # rmse_f_trn
                    except: pass

    # Find model
    model_path = os.path.join(work_dir, "model.ckpt.pt")
    if not os.path.exists(model_path):
        for fn in os.listdir(work_dir):
            if fn.endswith(".pt") and "model" in fn:
                model_path = os.path.join(work_dir, fn); break

    return model_path, train_losses


def run_optuna_fixed(data_dir, n_trials=6):
    print("\n" + "="*70)
    print("  PHASE 3: DeepMD Training with Optuna HPO (CPU, sel=auto)")
    print(f"  Trials: {n_trials}")
    print("="*70)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_losses = []

    def objective(trial):
        params = {
            "rcut": trial.suggest_float("rcut", 5.0, 7.0, step=0.5),
            "rcut_smth": trial.suggest_float("rcut_smth", 0.3, 1.0, step=0.1),
            "neuron": trial.suggest_categorical("neuron_config", ["small", "medium", "large"]),
            "start_lr": trial.suggest_float("start_lr", 5e-4, 5e-3, log=True),
            "numb_steps": trial.suggest_categorical("numb_steps", [2000, 3000, 5000]),
            "decay_steps": trial.suggest_categorical("decay_steps", [300, 500]),
        }
        neuron_map = {"small": [16, 32, 64], "medium": [25, 50, 100], "large": [32, 64, 128]}
        fitting_map = {"small": [120, 120, 120], "medium": [240, 240, 240], "large": [240, 240, 240]}
        nk = params["neuron"]
        params["neuron"] = neuron_map[nk]
        params["fitting_neuron"] = fitting_map[nk]

        trial_dir = os.path.join(TRAIN_DIR, f"trial_fixed_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        try:
            input_path = create_deepmd_input(params, data_dir, trial_dir)
            t0 = time.time()
            model_path, train_l = train_deepmd_cpu(input_path, trial_dir)
            dt = time.time() - t0
            final_loss = train_l[-1] if train_l else float('inf')
            print(f"    Trial {trial.number}: {dt:.0f}s, final_rmse_f={final_loss:.6f}")
            all_losses.append({
                "trial": trial.number,
                "params": {k: v for k, v in trial.params.items()},
                "train_losses": train_l, "time": dt,
            })
            return final_loss
        except Exception as e:
            print(f"    Trial {trial.number} FAILED: {e}")
            traceback.print_exc()
            return float('inf')

    study = optuna.create_study(direction="minimize", study_name="deepmd_fixed")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Best trial: {study.best_trial.number}")
    print(f"  Best params: {study.best_trial.params}")
    print(f"  Best loss: {study.best_trial.value:.6f}")

    optuna_results = {
        "best_trial": study.best_trial.number,
        "best_params": {k: v for k, v in study.best_trial.params.items()},
        "best_value": study.best_trial.value,
        "all_trials": all_losses,
    }
    with open(os.path.join(OUT, "optuna_results.json"), 'w') as f:
        json.dump(optuna_results, f, indent=2)

    best_dir = os.path.join(TRAIN_DIR, f"trial_fixed_{study.best_trial.number}")
    best_model = os.path.join(best_dir, "model.ckpt.pt")
    if not os.path.exists(best_model):
        for fn in os.listdir(best_dir):
            if fn.endswith(".pt") and "model" in fn:
                best_model = os.path.join(best_dir, fn); break

    return best_model, optuna_results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 3 Fix: DeepMD Optuna CPU Training (sel=auto)             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    data_dir = os.path.join(TRAIN_DIR, "data")
    assert os.path.exists(data_dir), f"Training data not found: {data_dir}"

    # Run Optuna
    best_model, optuna_res = run_optuna_fixed(data_dir, n_trials=6)
    
    if best_model and os.path.exists(best_model):
        print(f"\n  Best model: {best_model}")
        
        # Benchmark fine-tuned model
        from deepmd.calculator import DP as DPCalc
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Stay on CPU for inference too
        
        def dp_calc():
            return DPCalc(model=best_model)
        
        ft_results = {}
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
                
                ft_results[el] = {
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
        
        # Save & merge
        existing_path = os.path.join(OUT, "deepmd_meam_results.json")
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing = json.load(f)
        else:
            existing = {}
        
        existing["DeepMD-finetuned"] = ft_results
        with open(existing_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        # Merge with main
        main_path = os.path.join(OUT, "results.json")
        if os.path.exists(main_path):
            with open(main_path) as f:
                main = json.load(f)
            main.update(existing)
            with open(os.path.join(OUT, "all_results_merged.json"), 'w') as f:
                json.dump(main, f, indent=2)
            print(f"✓ All results merged")
    
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  COMPLETE                                                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
