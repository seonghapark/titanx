#!/usr/bin/env python3
"""Test that YaRN-scaled agpt configs instantiate correctly."""

import sys
from pathlib import Path

# Setup path
tt_root = Path("/lus/flare/projects/datascience/seonghapark/xpu_launcher/xpu_torchtitan/torchtitan_repo")
sys.path.insert(0, str(tt_root))

print("Testing YaRN-scaled agpt config instantiation...\n")

# Test 1: Import and list flavors
print("1. Import agpt_configs")
try:
    from torchtitan.models.agpt import agpt_configs
    print(f"   ✓ Successfully imported agpt_configs")
    print(f"   Total flavors: {len(agpt_configs)}")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check new flavors exist
print("\n2. Check YaRN flavors in registry")
for flavor in ["2B_yarn", "20B_yarn", "2b_yarn", "20b_yarn"]:
    if flavor in agpt_configs:
        print(f"   ✓ {flavor}")
    else:
        print(f"   ✗ {flavor} not found")
        sys.exit(1)

# Test 3: Verify config properties
print("\n3. Verify 2B_yarn config properties")
cfg = agpt_configs["2B_yarn"]
checks = [
    ("Type", type(cfg).__name__ == "Config", f"Config (got {type(cfg).__name__})"),
    ("Layers", len(cfg.layers) == 12, "12 layers"),
    ("Rope type", type(cfg.layers[0].attention.rope).__name__ == "Config", "RoPE.Config"),
    ("Rope scaling", cfg.layers[0].attention.rope.scaling == "yarn", "yarn"),
    ("Rope max_seq_len", cfg.layers[0].attention.rope.max_seq_len == 262144, "262144"),
    ("Rope rope_factor", cfg.layers[0].attention.rope.rope_factor == 32.0, "32.0"),
]

for check_name, condition, expected in checks:
    if condition:
        print(f"   ✓ {check_name:20} = {expected}")
    else:
        print(f"   ✗ {check_name:20} expected {expected}")
        sys.exit(1)

# Test 4: Verify config_registry functions
print("\n4. Verify config_registry functions")
try:
    from torchtitan.models.agpt.config_registry import (
        agpt_2b_yarn,
        agpt_20b_yarn,
        ezpz_agpt_2b_yarn,
        ezpz_agpt_20b_yarn,
    )
    print(f"   ✓ All 4 functions imported successfully")
    
    # Try calling them
    trainer_cfg_2b = agpt_2b_yarn(seq_len=16384)
    print(f"   ✓ agpt_2b_yarn(seq_len=16384) callable")
    
    trainer_cfg_20b = agpt_20b_yarn(seq_len=16384)
    print(f"   ✓ agpt_20b_yarn(seq_len=16384) callable")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All instantiation tests passed!")
print("\nYaRN-scaled agpt models are ready for training.")
