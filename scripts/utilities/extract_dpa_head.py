#!/usr/bin/env python3
"""
Extract DPA-2 Domains_SemiCond head by renaming state_dict keys.
The issue: deepmd expects model.Default.* but the pretrained model has model.Domains_SemiCond.*
Solution: Create a new checkpoint with the SemiCond keys renamed to Default.
"""
import sys, os, copy, json
log_f = open("/tmp/dpa_extract.log", "w")
sys.stdout = log_f
sys.stderr = log_f

import torch

BASE = "/home/bib569/Workspace/Hackrush_2026-Problem-10"

for model_name, model_file in [("dpa2_7M", "dpa-2.4-7M.pt"), ("dpa3_3M", "DPA-3.1-3M.pt")]:
    src = os.path.join(BASE, "potentials", model_file)
    dst = os.path.join(BASE, "potentials", f"{model_name}_semicond.pt")
    
    print(f"\n{'='*60}")
    print(f"Extracting Domains_SemiCond from {model_file}")
    print(f"{'='*60}")
    
    try:
        data = torch.load(src, map_location='cpu', weights_only=False)
        
        if not isinstance(data, dict):
            print(f"  Not a dict, type={type(data)}")
            continue
        
        new_data = {}
        
        # Copy all non-model keys as-is
        for k, v in data.items():
            if k != 'model':
                new_data[k] = v
        
        # Process model dict: keep only shared and Domains_SemiCond keys
        if 'model' in data and isinstance(data['model'], dict):
            old_model = data['model']
            new_model = {}
            
            # Find all unique head prefixes
            head_prefixes = set()
            for k in old_model:
                parts = k.split(".")
                if len(parts) >= 2 and parts[0] == "model":
                    head_prefixes.add(parts[1])
            print(f"  Found head prefixes: {sorted(head_prefixes)[:10]}...")
            
            # Strategy: rename model.Domains_SemiCond.* -> model.Default.*
            # and keep everything that's NOT model.<other_head>.*
            renamed = 0
            kept = 0
            dropped = 0
            
            for k, v in old_model.items():
                if k.startswith("model.Domains_SemiCond."):
                    # Rename to Default
                    new_key = "model.Default." + k[len("model.Domains_SemiCond."):]
                    new_model[new_key] = v
                    renamed += 1
                elif k.startswith("model."):
                    parts = k.split(".")
                    if len(parts) >= 2 and parts[1] in head_prefixes and parts[1] != "Domains_SemiCond":
                        # This is another head's key - drop it
                        dropped += 1
                    else:
                        # Shared key
                        new_model[k] = v
                        kept += 1
                else:
                    new_model[k] = v
                    kept += 1
            
            print(f"  Renamed: {renamed}, Kept: {kept}, Dropped: {dropped}")
            new_data['model'] = new_model
            
            # Also need to update model_def_script or config if present
            # Check for model config keys
            for ck in ['model_def_script', '@variables']:
                if ck in new_data:
                    print(f"  Has {ck}")
            
            # We may need to modify the model params to indicate single-task
            if '@variables' in new_data:
                vd = new_data['@variables']
                # Check for multi-task related variables
                for vk in list(vd.keys()):
                    if 'Domains_SemiCond' in vk:
                        new_vk = vk.replace('Domains_SemiCond', 'Default')
                        vd[new_vk] = vd.pop(vk)
                    elif any(hp in vk for hp in head_prefixes if hp != 'Domains_SemiCond'):
                        del vd[vk]
                new_data['@variables'] = vd
        
        # Save
        torch.save(new_data, dst)
        print(f"  Saved to {dst}")
        print(f"  Size: {os.path.getsize(dst)} bytes")
        
        # Test loading
        print("  Testing load...")
        try:
            from deepmd.calculator import DP as DPCalc
            calc = DPCalc(model=dst)
            print("  SUCCESS loading without head!")
            
            from ase.build import bulk
            at = bulk("Si", crystalstructure='diamond', a=5.431)
            at.calc = calc
            e = at.get_potential_energy()
            f = at.get_forces()
            print(f"  Si energy: {e:.6f} eV, max force: {abs(f).max():.6f}")
            
            at2 = bulk("Ge", crystalstructure='diamond', a=5.658)
            at2.calc = DPCalc(model=dst)
            e2 = at2.get_potential_energy()
            f2 = at2.get_forces()
            print(f"  Ge energy: {e2:.6f} eV, max force: {abs(f2).max():.6f}")
            print(f"  *** {model_name} WORKING! ***")
        except Exception as ex:
            print(f"  Load failed: {ex}")
            err = str(ex)
            if "Missing key" in err:
                # Find which keys are missing
                import re
                missing = re.findall(r'"([^"]+)"', err[:2000])
                print(f"  Missing keys (first 10): {missing[:10]}")
            
            # Try with head=Default
            print("  Trying with head=Default...")
            try:
                calc2 = DPCalc(model=dst, head="Default")
                at = bulk("Si", crystalstructure='diamond', a=5.431)
                at.calc = calc2
                e = at.get_potential_energy()
                print(f"  Si energy (head=Default): {e:.6f} eV")
                print(f"  *** {model_name} WORKING with head=Default! ***")
            except Exception as ex2:
                print(f"  head=Default also failed: {ex2}")
                
    except Exception as ex:
        print(f"  ERROR: {ex}")
        import traceback
        traceback.print_exc()

log_f.flush()
log_f.close()
