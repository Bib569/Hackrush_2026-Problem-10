#!/usr/bin/env python3
"""
Full Benchmark: MACE, CHGNet, Tersoff, SW for Si & Ge
Computes: lattice constant (EOS), cohesive energy, forces (bootstrap)
Generates: publication-quality parity/correlation/convergence plots
"""
import os, sys, json, csv, subprocess, tempfile, shutil, warnings, traceback
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LinearRegression
from ase.build import bulk
from ase.optimize import BFGS
from ase.eos import EquationOfState
from ase.io import write as ase_write
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
warnings.filterwarnings('ignore')

BASE = "/home/bib569/Workspace/Hackrush_2026-Problem-10"
OUT = os.path.join(BASE, "results_comprehensive")
POT = os.path.join(BASE, "potentials")
LMP = "/usr/bin/lmp"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'axes.linewidth':1.2,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':9,'figure.titlesize':14,'axes.titlesize':12,'lines.linewidth':2,
    'lines.markersize':6,'figure.dpi':150})

GT = {
    "Si": {"exp_a":5.431,"dft_a":5.469,"exp_e":-4.63,"dft_e":-5.43},
    "Ge": {"exp_a":5.658,"dft_a":5.763,"exp_e":-3.85,"dft_e":-4.62},
}
MASS = {"Si":28.085,"Ge":72.630}
COLORS = {"MACE-MP-0":"#E63946","CHGNet":"#457B9D","Tersoff":"#E9C46A","SW":"#F4A261"}
MARKERS = {"MACE-MP-0":"o","CHGNet":"s","Tersoff":"^","SW":"v"}

# ── LAMMPS subprocess calculator ──
def lammps_energy_forces(atoms, pair_style, pair_coeff, element):
    d = tempfile.mkdtemp(prefix="lmp_")
    try:
        ase_write(os.path.join(d,"s.data"), atoms, format="lammps-data")
        s = f"units metal\natom_style atomic\nboundary p p p\nbox tilt large\nread_data s.data\nmass 1 {MASS[element]}\npair_style {pair_style}\npair_coeff {pair_coeff}\nthermo_style custom step pe\nthermo 1\ndump 1 all custom 1 f.dump id fx fy fz\ndump_modify 1 sort id format float \"%20.12f\"\nrun 0\n"
        with open(os.path.join(d,"in.x"),'w') as f: f.write(s)
        r = subprocess.run([LMP,"-in","in.x"], cwd=d, capture_output=True, text=True, timeout=30)
        if r.returncode!=0: raise RuntimeError(r.stdout[-500:])
        energy=None
        with open(os.path.join(d,"log.lammps")) as f:
            th=False
            for line in f:
                if "Step" in line and "PotEng" in line: th=True; continue
                if th and ("Loop" in line): th=False; continue
                if th:
                    p=line.strip().split()
                    if len(p)>=2:
                        try: energy=float(p[1])
                        except: pass
        n=len(atoms); forces=np.zeros((n,3))
        with open(os.path.join(d,"f.dump")) as f: lines=f.readlines()
        rd=False
        for line in lines:
            if "ITEM: ATOMS" in line: rd=True; continue
            if rd:
                p=line.strip().split()
                if len(p)>=4: forces[int(p[0])-1]=[float(p[1]),float(p[2]),float(p[3])]
        return energy, forces
    finally:
        shutil.rmtree(d, ignore_errors=True)

class LammpsCalc:
    """ASE-like calculator wrapper for LAMMPS subprocess."""
    def __init__(self, pair_style, pair_coeff, element):
        self.ps, self.pc, self.el = pair_style, pair_coeff, element
        self._e, self._f = None, None
    def _calc(self, atoms):
        self._e, self._f = lammps_energy_forces(atoms, self.ps, self.pc, self.el)
    def get_potential_energy(self, atoms):
        self._calc(atoms); return self._e
    def get_forces(self, atoms):
        if self._f is None: self._calc(atoms)
        return self._f

# ── ASE-compatible wrapper ──
class CalcWrapper:
    """Wraps ASE or LAMMPS calc for uniform interface."""
    def __init__(self, calc_fn, is_lammps=False, pair_style="", pair_coeff_map=None):
        self.calc_fn=calc_fn; self.is_lammps=is_lammps
        self.ps=pair_style; self.pc_map=pair_coeff_map or {}
    def energy_forces(self, atoms, element):
        if self.is_lammps:
            pc = self.pc_map.get(element)
            if pc is None:
                raise ValueError(f"No pair_coeff for {element} in {self.ps}")
            e,f = lammps_energy_forces(atoms, self.ps, pc, element)
            return e, f
        else:
            a = atoms.copy()
            a.calc = self.calc_fn()
            return a.get_potential_energy(), a.get_forces()

# ── Benchmark functions ──
def eos_lattice(wrapper, element, n=11, sr=0.06):
    a0 = GT[element]["exp_a"]
    strains = np.linspace(1-sr,1+sr,n)
    vols,ens,avals=[],[],[]
    for s in strains:
        at = bulk(element, crystalstructure='diamond', a=a0*s)
        e,_ = wrapper.energy_forces(at, element)
        vols.append(at.get_volume()); ens.append(e); avals.append(a0*s)
    V,E,A = np.array(vols),np.array(ens),np.array(avals)
    try:
        eos=EquationOfState(V,E,eos='birchmurnaghan'); v0,e0,B=eos.fit()
        a_eq=(v0*4.0)**(1./3.)
    except:
        idx=np.argmin(E); a_eq=A[idx]; e0=E[idx]
    return a_eq, e0/2.0, A, E/2.0

def relax_cell(wrapper, element):
    if wrapper.is_lammps:
        # For LAMMPS: scan finely around equilibrium
        a0=GT[element]["exp_a"]; best_e=1e10; best_a=a0
        for s in np.linspace(0.97,1.03,31):
            at=bulk(element,crystalstructure='diamond',a=a0*s)
            e,_=wrapper.energy_forces(at,element)
            if e<best_e: best_e=e; best_a=a0*s
        at=bulk(element,crystalstructure='diamond',a=best_a)
        _,f=wrapper.energy_forces(at,element)
        return best_a, best_e/2.0, np.max(np.abs(f))
    else:
        at=bulk(element,crystalstructure='diamond',a=GT[element]["exp_a"])
        at.calc=wrapper.calc_fn()
        ecf=ExpCellFilter(at); opt=BFGS(ecf,logfile=None)
        try: opt.run(fmax=0.01,steps=200)
        except: pass
        cell=at.get_cell()
        a_r=np.mean([np.linalg.norm(cell[i]) for i in range(3)])*(2**0.5)
        return a_r, at.get_potential_energy()/len(at), np.max(np.abs(at.get_forces()))

def bootstrap_forces(wrapper, element, n=30, seed=42):
    rng=np.random.RandomState(seed); disps=[]; fmags=[]
    a0=GT[element]["exp_a"]
    for i in range(n):
        at=bulk(element,crystalstructure='diamond',a=a0)
        dm=rng.uniform(0.01,0.2)
        at.set_positions(at.get_positions()+rng.randn(*at.get_positions().shape)*dm)
        _,f=wrapper.energy_forces(at,element)
        disps.append(dm); fmags.append(f.flatten().tolist())
    return np.array(disps), fmags

def energy_strain(wrapper, element, n=21):
    a0=GT[element]["exp_a"]; strains=np.linspace(0.85,1.15,n); Es=[]
    for s in strains:
        at=bulk(element,crystalstructure='diamond',a=a0*s)
        e,_=wrapper.energy_forces(at,element); Es.append(e/2.0)
    return strains*a0, np.array(Es)

# ── Load all models ──
def load_models():
    models = {}
    # MACE
    try:
        from mace.calculators import mace_mp
        _m=mace_mp(model="medium",dispersion=False,default_dtype="float32")
        models["MACE-MP-0"]=CalcWrapper(lambda c=_m:c)
        print("  ✓ MACE-MP-0")
    except Exception as e: print(f"  ✗ MACE: {e}")
    # CHGNet
    try:
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        cm=CHGNet.load()
        models["CHGNet"]=CalcWrapper(lambda m=cm:CHGNetCalculator(model=m))
        print("  ✓ CHGNet")
    except Exception as e: print(f"  ✗ CHGNet: {e}")
    # Tersoff
    try:
        tersoff_pc = {
            "Si": f"* * {POT}/Si.tersoff Si",
            "Ge": f"* * {POT}/SiCGe.tersoff Ge",
        }
        # Quick test
        at=bulk("Si",crystalstructure='diamond',a=5.43)
        lammps_energy_forces(at,"tersoff",tersoff_pc["Si"],"Si")
        models["Tersoff"]=CalcWrapper(None,is_lammps=True,pair_style="tersoff",
            pair_coeff_map=tersoff_pc)
        print("  ✓ Tersoff (LAMMPS)")
    except Exception as e: print(f"  ✗ Tersoff: {e}")
    # SW
    try:
        sw_pc = {
            "Si": f"* * {POT}/Si.sw Si",
            "Ge": f"* * {POT}/Ge.sw Ge",
        }
        at=bulk("Si",crystalstructure='diamond',a=5.43)
        lammps_energy_forces(at,"sw",sw_pc["Si"],"Si")
        models["SW"]=CalcWrapper(None,is_lammps=True,pair_style="sw",
            pair_coeff_map=sw_pc)
        print("  ✓ SW (LAMMPS) — Si + Ge")
    except Exception as e: print(f"  ✗ SW: {e}")
    return models

# ── Main benchmark ──
def run_benchmarks():
    print("="*70+"\nLoading models...\n"+"="*70)
    models = load_models()
    results = {}

    for name in models:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        results[name] = {}
        w = models[name]

        elements = ["Si","Ge"]

        for el in elements:
            print(f"\n  --- {el} ---")
            try:

                # EOS
                print("  [1] EOS...")
                a_eos, e_eos, a_crv, e_crv = eos_lattice(w, el)
                print(f"      a={a_eos:.4f} Å (DFT={GT[el]['dft_a']:.3f})")

                # Relax
                print("  [2] Relax...")
                a_rlx, e_rlx, mf = relax_cell(w, el)
                print(f"      a={a_rlx:.4f}, E/at={e_rlx:.4f}, maxF={mf:.2e}")

                # Bootstrap forces
                print("  [3] Bootstrap forces (30 configs)...")
                disps, fmags = bootstrap_forces(w, el)
                max_fs = [np.max(np.abs(np.array(f).reshape(-1,3))) for f in fmags]
                print(f"      <maxF>={np.mean(max_fs):.4f}±{np.std(max_fs):.4f}")

                # E-strain
                print("  [4] E-strain...")
                s_a, s_e = energy_strain(w, el)

                results[name][el] = {
                    "a_eos":float(a_eos),"a_rlx":float(a_rlx),
                    "e_eos":float(e_eos),"e_rlx":float(e_rlx),
                    "maxf_rlx":float(mf),
                    "mean_maxf":float(np.mean(max_fs)),"std_maxf":float(np.std(max_fs)),
                    "eos_a":a_crv.tolist(),"eos_e":e_crv.tolist(),
                    "strain_a":s_a.tolist(),"strain_e":s_e.tolist(),
                    "force_disps":disps.tolist(),"force_mags":fmags,
                }
                print(f"  ✓ {name}/{el} done")
            except Exception as ex:
                print(f"  ✗ {name}/{el}: {ex}"); traceback.print_exc()

    with open(os.path.join(OUT,"results.json"),'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n✓ Saved {OUT}/results.json")
    return results

# ══════════ PLOTTING ══════════

def ci_band(x,y):
    lr=LinearRegression(); lr.fit(x.reshape(-1,1),y)
    yp=lr.predict(x.reshape(-1,1)); n=len(x)
    if n<=2: return yp,yp,yp
    se=np.sqrt(np.sum((y-yp)**2)/(n-2))
    tv=stats.t.ppf(0.975,n-2)
    ci=tv*se*np.sqrt(1+1/n+(x-np.mean(x))**2/np.sum((x-np.mean(x))**2))
    return yp,yp-ci,yp+ci

def plot_all(results):
    mnames=list(results.keys())
    els=["Si","Ge"]

    # ── Fig 1: Combined Correlation (3-panel) ──
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    fig.suptitle("Correlation Plots: Predicted vs Ground Truth",fontsize=14,fontweight='bold')
    props = [
        ("Lattice Constant","dft_a","a_eos","Å"),
        ("Cohesive Energy","dft_e","e_rlx","eV/atom"),
        ("EOS Energy","dft_e","e_eos","eV/atom"),
    ]
    sublbl=['a','b','c']
    pcols=['indigo','forestgreen','crimson']
    pmkrs=['o','s','^']
    for pi,(pname,gk,pk,unit) in enumerate(props):
        ax=axes[pi]; gt_v=[]; pr_v=[]; lbls=[]
        for m in mnames:
            for el in els:
                if el in results.get(m,{}):
                    gt_v.append(GT[el][gk]); pr_v.append(results[m][el][pk])
                    lbls.append(f"{m}\n({el})")
        gt_v,pr_v=np.array(gt_v),np.array(pr_v)
        ax.scatter(gt_v,pr_v,c=pcols[pi],s=80,marker=pmkrs[pi],edgecolors='k',zorder=5,alpha=0.8)
        si=np.argsort(gt_v); yp,lo,hi=ci_band(gt_v[si],pr_v[si])
        ax.plot(gt_v[si],yp,'r-',lw=2,label='LinReg')
        ax.fill_between(gt_v[si],lo,hi,color='lightblue',alpha=0.3,label='95% CI')
        vv=np.concatenate([gt_v,pr_v]); vm,vx=vv.min()-0.15,vv.max()+0.15
        ax.plot([vm,vx],[vm,vx],'k--',alpha=0.4,lw=1)
        r2=np.corrcoef(gt_v,pr_v)[0,1]**2 if len(gt_v)>1 else 0
        mae=np.mean(np.abs(gt_v-pr_v)); rmse=np.sqrt(np.mean((gt_v-pr_v)**2))
        ax.text(0.05,0.95,f'R²={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}',
               transform=ax.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
        for i,l in enumerate(lbls):
            ax.annotate(l,(gt_v[i],pr_v[i]),textcoords="offset points",xytext=(6,-10),fontsize=6)
        ax.text(-0.1,1.02,f'({sublbl[pi]})',transform=ax.transAxes,fontsize=18,fontweight='bold')
        ax.set_xlabel(f"DFT {pname} ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
        ax.legend(fontsize=8,loc='lower right'); ax.grid(True,alpha=0.3,ls='--')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,"combined_correlation_plots.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ combined_correlation_plots.png")

    # ── Fig 2: Per-model lattice parity ──
    nm=len(mnames); nc=min(3,nm); nr=(nm+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(5.5*nc,5*nr),squeeze=False)
    fig.suptitle("Per-Model Correlation: Lattice Constant",fontsize=14,fontweight='bold',y=1.01)
    for i,m in enumerate(mnames):
        ax=axes[i//nc,i%nc]; c=COLORS.get(m,'gray'); mk=MARKERS.get(m,'o')
        gv,pv,lb=[],[],[]
        for el in els:
            if el in results.get(m,{}):
                gv.append(GT[el]["dft_a"]); pv.append(results[m][el]["a_eos"]); lb.append(el)
        if not pv: continue
        gv,pv=np.array(gv),np.array(pv)
        ax.scatter(gv,pv,c=c,s=120,marker=mk,edgecolors='k',zorder=5)
        for k,l in enumerate(lb): ax.annotate(l,(gv[k],pv[k]),xytext=(8,5),textcoords="offset points",fontsize=11,fontweight='bold')
        av=np.concatenate([gv,pv]); vm,vx=av.min()-0.15,av.max()+0.15
        ax.plot([vm,vx],[vm,vx],'k--',alpha=0.5); ax.set_xlim(vm,vx); ax.set_ylim(vm,vx); ax.set_aspect('equal')
        if len(pv)>=2:
            r2=np.corrcoef(gv,pv)[0,1]**2; mae=np.mean(np.abs(gv-pv))
            ax.text(0.05,0.92,f'R²={r2:.4f}\nMAE={mae:.4f} Å',transform=ax.transAxes,fontsize=9,va='top',
                   bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
        ax.set_xlabel("DFT a₀ (Å)"); ax.set_ylabel("Pred a₀ (Å)")
        ax.set_title(m,fontweight='bold'); ax.grid(True,ls='--',alpha=0.3)
    for i in range(nm,nr*nc): axes[i//nc,i%nc].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,"correlation_lattice_per_model.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ correlation_lattice_per_model.png")

    # ── Fig 3: Energy convergence ──
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Energy Convergence (EOS sampling density)",fontsize=14,fontweight='bold')
    for ei,el in enumerate(els):
        ax=axes[ei]
        for m in mnames:
            if el not in results.get(m,{}): continue
            ac=np.array(results[m][el]["eos_a"]); ec=np.array(results[m][el]["eos_e"])
            np_=len(ac); cn=[]; ca=[]
            for n in range(3,np_+1):
                idx=np.round(np.linspace(0,np_-1,n)).astype(int)
                sv=((ac[idx])**3)/4.0; se=ec[idx]*2
                try:
                    eos=EquationOfState(sv,se,eos='birchmurnaghan'); v0,_,_=eos.fit()
                    cn.append(n); ca.append((v0*4)**(1./3.))
                except: pass
            if cn: ax.plot(cn,ca,'-o',color=COLORS.get(m,'gray'),label=m,ms=4)
        ax.axhline(GT[el]["dft_a"],color='purple',ls=':',lw=1.5,label=f'DFT ({GT[el]["dft_a"]})')
        ax.axhline(GT[el]["exp_a"],color='green',ls='--',lw=1.5,label=f'Exp ({GT[el]["exp_a"]})')
        ax.set_xlabel("N EOS points"); ax.set_ylabel("Predicted a₀ (Å)")
        ax.set_title(el,fontweight='bold'); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,"energy_convergence.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ energy_convergence.png")

    # ── Fig 4: Force bootstrap CV ──
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Bootstrap Force Cross-Validation",fontsize=14,fontweight='bold')
    for ei,el in enumerate(els):
        ax=axes[ei]
        for m in mnames:
            if el not in results.get(m,{}): continue
            d=results[m][el]; ds=np.array(d.get("force_disps",[])); fm=d.get("force_mags",[])
            if len(ds)==0: continue
            mxf=np.array([np.max(np.abs(np.array(f).reshape(-1,3))) for f in fm])
            ax.scatter(ds,mxf,c=COLORS.get(m,'gray'),s=25,alpha=0.6,marker=MARKERS.get(m,'o'),
                      edgecolors='k',lw=0.3,label=m)
            if len(ds)>3:
                z=np.polyfit(ds,mxf,2); p=np.poly1d(z); xs=np.linspace(ds.min(),ds.max(),50)
                ax.plot(xs,p(xs),'-',color=COLORS.get(m,'gray'),alpha=0.7)
        ax.set_xlabel("Displacement (Å)"); ax.set_ylabel("Max |Force| (eV/Å)")
        ax.set_title(el,fontweight='bold'); ax.legend(fontsize=8,ncol=2); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,"force_cv_bootstrap.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ force_cv_bootstrap.png")

    # ── Fig 5: E-strain curves ──
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Energy vs Lattice Constant",fontsize=14,fontweight='bold')
    for ei,el in enumerate(els):
        ax=axes[ei]
        for m in mnames:
            if el not in results.get(m,{}): continue
            ax.plot(results[m][el]["strain_a"],results[m][el]["strain_e"],'-',
                   color=COLORS.get(m,'gray'),lw=2,label=m)
        ax.axvline(GT[el]["exp_a"],color='green',ls='--',lw=1.5,label=f'Exp')
        ax.axvline(GT[el]["dft_a"],color='purple',ls=':',lw=1.5,label=f'DFT')
        ax.set_xlabel("a (Å)"); ax.set_ylabel("E/atom (eV)")
        ax.set_title(el,fontweight='bold'); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,"energy_strain_curves.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ energy_strain_curves.png")

    # ── Fig 6: Error comparison bar chart ──
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Prediction Errors vs Ground Truth",fontsize=14,fontweight='bold')
    # Lattice
    ax=axes[0]; xp=np.arange(2); w=0.8/max(len(mnames),1)
    for mi,m in enumerate(mnames):
        errs=[]
        for el in els:
            if el in results.get(m,{}):
                errs.append(abs(results[m][el]["a_eos"]-GT[el]["dft_a"])/GT[el]["dft_a"]*100)
            else: errs.append(0)
        off=(mi-len(mnames)/2+0.5)*w
        bars=ax.bar(xp+off,errs,w*0.9,color=COLORS.get(m,'gray'),edgecolor='k',lw=0.5,label=m)
        for b,e in zip(bars,errs):
            if e>0: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{e:.2f}%',ha='center',fontsize=7)
    ax.set_xticks(xp); ax.set_xticklabels(els); ax.set_ylabel("Error vs DFT (%)")
    ax.set_title("Lattice Constant",fontweight='bold'); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3,axis='y')
    # Energy
    ax=axes[1]
    for mi,m in enumerate(mnames):
        errs=[]
        for el in els:
            if el in results.get(m,{}):
                errs.append(abs(results[m][el]["e_rlx"]-GT[el]["dft_e"]))
            else: errs.append(0)
        off=(mi-len(mnames)/2+0.5)*w
        bars=ax.bar(xp+off,errs,w*0.9,color=COLORS.get(m,'gray'),edgecolor='k',lw=0.5,label=m)
        for b,e in zip(bars,errs):
            if e>0: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{e:.3f}',ha='center',fontsize=7)
    ax.set_xticks(xp); ax.set_xticklabels(els); ax.set_ylabel("|Error| (eV/atom)")
    ax.set_title("Cohesive Energy",fontweight='bold'); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3,axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,"error_comparison.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ error_comparison.png")

    # ── Fig 7: Comprehensive 6-panel ──
    fig=plt.figure(figsize=(20,13))
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.35,wspace=0.35)
    fig.suptitle("Comprehensive MLIP & Classical Potential Benchmark (Si & Ge)",fontsize=15,fontweight='bold',y=0.99)
    # (a) lattice parity
    ax=fig.add_subplot(gs[0,0])
    for m in mnames:
        for el in els:
            if el not in results.get(m,{}): continue
            ax.scatter(GT[el]["dft_a"],results[m][el]["a_eos"],c=COLORS.get(m,'gray'),
                      s=120,marker=MARKERS.get(m,'o'),edgecolors='k',zorder=5)
            ax.annotate(el,(GT[el]["dft_a"],results[m][el]["a_eos"]),xytext=(6,4),textcoords="offset points",fontsize=8)
    av=[]
    for m in mnames:
        for el in els:
            if el in results.get(m,{}): av.extend([GT[el]["dft_a"],results[m][el]["a_eos"]])
    if av: vm,vx=min(av)-0.1,max(av)+0.1; ax.plot([vm,vx],[vm,vx],'k--',alpha=0.5); ax.set_xlim(vm,vx); ax.set_ylim(vm,vx)
    ax.set_aspect('equal'); ax.set_xlabel("DFT a₀"); ax.set_ylabel("Pred a₀"); ax.set_title("(a) Lattice vs DFT",fontweight='bold'); ax.grid(True,alpha=0.3)
    # (b) energy parity
    ax=fig.add_subplot(gs[0,1])
    for m in mnames:
        for el in els:
            if el not in results.get(m,{}): continue
            ax.scatter(GT[el]["dft_e"],results[m][el]["e_rlx"],c=COLORS.get(m,'gray'),
                      s=120,marker=MARKERS.get(m,'o'),edgecolors='k',zorder=5)
            ax.annotate(el,(GT[el]["dft_e"],results[m][el]["e_rlx"]),xytext=(6,4),textcoords="offset points",fontsize=8)
    av=[]
    for m in mnames:
        for el in els:
            if el in results.get(m,{}): av.extend([GT[el]["dft_e"],results[m][el]["e_rlx"]])
    if av: vm,vx=min(av)-0.3,max(av)+0.3; ax.plot([vm,vx],[vm,vx],'k--',alpha=0.5); ax.set_xlim(vm,vx); ax.set_ylim(vm,vx)
    ax.set_aspect('equal'); ax.set_xlabel("DFT E (eV/at)"); ax.set_ylabel("Pred E"); ax.set_title("(b) Energy vs DFT",fontweight='bold'); ax.grid(True,alpha=0.3)
    # (c) error bars
    ax=fig.add_subplot(gs[0,2])
    xp=np.arange(2); w=0.8/max(len(mnames),1)
    for mi,m in enumerate(mnames):
        errs=[abs(results[m].get(el,{}).get("a_eos",GT[el]["dft_a"])-GT[el]["dft_a"])/GT[el]["dft_a"]*100 if el in results.get(m,{}) else 0 for el in els]
        off=(mi-len(mnames)/2+0.5)*w
        ax.bar(xp+off,errs,w*0.9,color=COLORS.get(m,'gray'),edgecolor='k',lw=0.5,label=m)
    ax.set_xticks(xp); ax.set_xticklabels(els); ax.set_ylabel("Error %"); ax.set_title("(c) a₀ Error",fontweight='bold')
    ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3,axis='y')
    # (d) Si E-strain
    ax=fig.add_subplot(gs[1,0])
    for m in mnames:
        if "Si" in results.get(m,{}): ax.plot(results[m]["Si"]["strain_a"],results[m]["Si"]["strain_e"],'-',color=COLORS.get(m,'gray'),lw=2,label=m)
    ax.axvline(GT["Si"]["exp_a"],color='green',ls='--'); ax.axvline(GT["Si"]["dft_a"],color='purple',ls=':')
    ax.set_xlabel("a (Å)"); ax.set_ylabel("E/at (eV)"); ax.set_title("(d) Si E-strain",fontweight='bold')
    ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)
    # (e) Ge E-strain
    ax=fig.add_subplot(gs[1,1])
    for m in mnames:
        if "Ge" in results.get(m,{}): ax.plot(results[m]["Ge"]["strain_a"],results[m]["Ge"]["strain_e"],'-',color=COLORS.get(m,'gray'),lw=2,label=m)
    ax.axvline(GT["Ge"]["exp_a"],color='green',ls='--'); ax.axvline(GT["Ge"]["dft_a"],color='purple',ls=':')
    ax.set_xlabel("a (Å)"); ax.set_ylabel("E/at (eV)"); ax.set_title("(e) Ge E-strain",fontweight='bold')
    ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)
    # (f) Force bootstrap
    ax=fig.add_subplot(gs[1,2])
    bd,bl,bc,be=[],[],[],[]
    for m in mnames:
        for el in els:
            if el in results.get(m,{}):
                bd.append(results[m][el].get("mean_maxf",0)); be.append(results[m][el].get("std_maxf",0))
                bl.append(f"{m}\n({el})"); bc.append(COLORS.get(m,'gray'))
    if bd:
        x=np.arange(len(bd)); ax.bar(x,bd,yerr=be,color=bc,edgecolor='k',lw=0.5,capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(bl,fontsize=7,rotation=45,ha='right')
    ax.set_ylabel("Mean Max|F| (eV/Å)"); ax.set_title("(f) Force Bootstrap",fontweight='bold'); ax.grid(True,alpha=0.3,axis='y')
    leg=[Patch(facecolor=COLORS.get(m,'gray'),edgecolor='k',label=m) for m in mnames]
    fig.legend(handles=leg,loc='lower center',ncol=len(mnames),fontsize=11,bbox_to_anchor=(0.5,-0.02))
    fig.savefig(os.path.join(OUT,"comprehensive_summary.png"),dpi=300,bbox_inches='tight',facecolor='w')
    plt.close(); print("  ✓ comprehensive_summary.png")

    # ── CSV table ──
    rows=[]
    for m in mnames:
        for el in els:
            if el not in results.get(m,{}): continue
            d=results[m][el]; g=GT[el]
            rows.append({"Model":m,"Type":"MLIP" if m in ["MACE-MP-0","CHGNet"] else "Classical",
                "Element":el,"a_EOS":f'{d["a_eos"]:.4f}',"a_Relax":f'{d["a_rlx"]:.4f}',
                "a_Exp":f'{g["exp_a"]:.3f}',"a_DFT":f'{g["dft_a"]:.3f}',
                "Err_Exp%":f'{abs(d["a_eos"]-g["exp_a"])/g["exp_a"]*100:.3f}',
                "Err_DFT%":f'{abs(d["a_eos"]-g["dft_a"])/g["dft_a"]*100:.3f}',
                "E_rlx":f'{d["e_rlx"]:.4f}',"E_DFT":f'{g["dft_e"]:.2f}',
                "MeanMaxF":f'{d.get("mean_maxf",0):.4f}',"StdMaxF":f'{d.get("std_maxf",0):.4f}'})
    if rows:
        with open(os.path.join(OUT,"summary.csv"),'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("  ✓ summary.csv")
    print("\n"+"-"*100)
    for r in rows:
        print(f"  {r['Model']:12s} {r['Type']:8s} {r['Element']:3s} | a_EOS={r['a_EOS']:>7s} a_DFT={r['a_DFT']:>6s} Err={r['Err_DFT%']:>6s}% | E={r['E_rlx']:>8s} E_DFT={r['E_DFT']:>6s} | <F>={r['MeanMaxF']:>7s}±{r['StdMaxF']}")
    print("-"*100)

if __name__=="__main__":
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  FULL BENCHMARK: MACE, CHGNet, Tersoff, SW for Si & Ge      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    results = run_benchmarks()
    print(f"\n{'='*60}\nGenerating Publication Plots\n{'='*60}")
    plot_all(results)
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  COMPLETE — Results in {OUT}")
    print("╚═══════════════════════════════════════════════════════════════╝")
