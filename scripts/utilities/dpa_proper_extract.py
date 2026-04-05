#!/usr/bin/env python3
"""
Properly extract DPA-2 Domains_SemiCond head into a single-task model.
Must modify both:
1. State dict keys: model.Domains_SemiCond.* -> model.Default.*
2. _extra_state.model_params.model_dict: keep only SemiCond entry as Default
"""
import sys, os, copy, json
log_f = open("/tmp/dpa_proper_extract.log", "w")
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
        model_dict = data['model']
        extra_state = model_dict['_extra_state']
        model_params = extra_state['model_params']
        
        # Step 1: Modify model_params - keep only SemiCond as 'Default'
        print("  Step 1: Modifying model_params...")
        old_model_dict_config = model_params['model_dict']
        head_names = list(old_model_dict_config.keys())
        print(f"  Original heads: {head_names[:10]}... ({len(head_names)} total)")
        
        if 'Domains_SemiCond' not in old_model_dict_config:
            print(f"  ERROR: Domains_SemiCond not in model_dict!")
            continue
        
        # Replace model_dict with single Default entry
        semicond_config = copy.deepcopy(old_model_dict_config['Domains_SemiCond'])
        new_model_dict_config = {'Default': semicond_config}
        model_params['model_dict'] = new_model_dict_config
        
        # Also update dim_case_embd to 0 since it's single-task now
        # (or keep it, let's try both approaches)
        
        # Step 2: Rename state_dict keys 
        print("  Step 2: Renaming state_dict keys...")
        new_state_dict = {}
        renamed = 0
        dropped = 0
        kept = 0
        
        # Get all head prefixes from state dict
        head_prefixes = set()
        for k in model_dict:
            if k == '_extra_state':
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "model":
                head_prefixes.add(parts[1])
        
        print(f"  State dict head prefixes: {sorted(head_prefixes)[:10]}...")
        
        for k, v in model_dict.items():
            if k == '_extra_state':
                # We'll add modified version later
                continue
            elif k.startswith("model.Domains_SemiCond."):
                new_key = "model.Default." + k[len("model.Domains_SemiCond."):]
                new_state_dict[new_key] = v
                renamed += 1
            elif k.startswith("model."):
                # Check if this is a shared weight (descriptor shared across heads)
                parts = k.split(".")
                if len(parts) >= 2 and parts[1] in head_prefixes:
                    # This belongs to another head - drop it
                    dropped += 1
                else:
                    # Shared parameter
                    new_state_dict[k] = v
                    kept += 1
            else:
                new_state_dict[k] = v
                kept += 1
        
        # Add modified extra_state
        new_state_dict['_extra_state'] = extra_state
        
        print(f"  Renamed: {renamed}, Kept: {kept}, Dropped: {dropped}")
        
        # Step 3: Build new checkpoint
        new_data = {
            'model': new_state_dict,
        }
        # Don't include optimizer state to save space
        
        # Save
        torch.save(new_data, dst)
        print(f"  Saved to {dst}")
        print(f"  Size: {os.path.getsize(dst)} bytes")
        
        # Step 4: Test loading
        print("  Step 4: Testing load...")
        try:
            from deepmd.calculator import DP as DPCalc
            calc = DPCalc(model=dst)
            print("  ✓ SUCCESS loading without head!")
            
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
            print(f"  Load test failed: {ex}")
            
            # It might still indicate multitask. Let's check if we need to
            # also set a flag to indicate single-task
            err = str(ex)
            if "multitask" in err.lower() or "head" in err.lower():
                print("  Still detected as multitask. Trying with head='Default'...")
                try:
                    calc = DPCalc(model=dst, head="Default")
                    at = bulk("Si", crystalstructure='diamond', a=5.431)
                    at.calc = calc
                    e = at.get_potential_energy()
                    print(f"  Si energy (head=Default): {e:.6f} eV")
                    print(f"  *** {model_name} WORKING with head=Default! ***")
                except Exception as ex2:
                    print(f"  head=Default also failed: {ex2}")
                    # Last resort: check what the error says about missing keys
                    if "Missing key" in str(ex2):
                        import re
                        missing = re.findall(r'"([^"]+)"', str(ex2)[:3000])
                        print(f"  Missing keys (first 5): {missing[:5]}")
                        # Check if the missing keys are in our new_state_dict
                        for mk in missing[:5]:
                            print(f"    '{mk}' in state_dict: {mk in new_state_dict}")
            elif "Missing key" in err:
                import re
                missing = re.findall(r'"([^"]+)"', err[:3000])
                print(f"  Missing keys (first 5): {missing[:5]}")
                # Check what keys we have that are similar
                for mk in missing[:5]:
                    # Find closest match 
                    candidates = [k for k in new_state_dict if mk.split(".")[-1] in k]
                    print(f"    '{mk}' candidates: {candidates[:3]}")
                    
    except Exception as ex:
        print(f"  FATAL ERROR: {ex}")
        import traceback
        traceback.print_exc()

log_f.flush()
log_f.close()
