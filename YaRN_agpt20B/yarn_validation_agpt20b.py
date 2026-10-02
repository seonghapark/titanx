#!/usr/bin/env python3
"""
YaRN validation script for agpt-20b-v2-512n-step-4000

This script validates YaRN configuration and demonstrates the rope scaling
mechanism for context extension from 8K to longer sequences.
"""

import json
import math
from typing import Tuple, List, Dict, Any

# YaRN Configuration (from config.json)
AGPT_20B_CONFIG = {
    "hidden_size": 5120,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "rope_theta": 500000.0,
    "max_position_embeddings": 131072,
    "rope_scaling": {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 1.0,
        "beta_slow": 32.0,
    }
}

def validate_yarn_parameters(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate YaRN parameters against mathematical constraints."""
    rope_scaling = config.get("rope_scaling", {})

    if rope_scaling.get("type") != "yarn":
        return False, "rope_scaling type must be 'yarn'"

    rope_factor = rope_scaling.get("factor", 1.0)
    original_seq_len = rope_scaling.get("original_max_position_embeddings", 2048)
    beta_fast = rope_scaling.get("beta_fast", 32.0)
    beta_slow = rope_scaling.get("beta_slow", 1.0)

    # Calculate head dimension
    d_head = config["hidden_size"] // config["num_attention_heads"]
    d_half = d_head // 2

    # Calculate frequency cutoff (YaRN paper: dimension where interpolation begins)
    rope_theta = config["rope_theta"]

    # The constraint is: 0 < low_freq_cutoff < high_freq_cutoff < d_half - 1
    # For YaRN with NTK-by-parts, the interpolation range must fit within d_half

    try:
        # Calculate correction dimensions (from YaRN reference implementation)
        # These define where the transition between low-freq and high-freq happens
        correction_dim = math.ceil(
            d_half * math.log(rope_factor) / math.log(rope_theta)
        )

        if not (0 < correction_dim < d_half - 1):
            return False, (
                f"correction_dim={correction_dim} violates constraint "
                f"0 < correction_dim < d_half-1 (d_half={d_half}). "
                f"Try adjusting rope_factor or model dimensions."
            )

        # Validate beta parameters
        if beta_fast <= 0 or beta_slow <= 0:
            return False, "beta_fast and beta_slow must be positive"

        if beta_fast >= beta_slow:
            return False, (
                f"beta_fast ({beta_fast}) should be < beta_slow ({beta_slow}). "
                "Per YaRN paper: fast=1-2 (interpolate), slow=32-40 (extrapolate)"
            )

        return True, "All parameters valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def calculate_yarn_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate YaRN-specific metrics."""
    rope_scaling = config.get("rope_scaling", {})
    rope_factor = rope_scaling.get("factor", 1.0)
    original_seq_len = rope_scaling.get("original_max_position_embeddings", 2048)
    beta_fast = rope_scaling.get("beta_fast", 32.0)
    beta_slow = rope_scaling.get("beta_slow", 1.0)

    d_head = config["hidden_size"] // config["num_attention_heads"]
    rope_theta = config["rope_theta"]
    max_pos = config["max_position_embeddings"]

    # Attention scaling correction from YaRN paper (eq. 4)
    mscale = 0.1 * math.log(rope_factor) + 1.0

    # Extended context after YaRN scaling
    extended_seq_len = int(original_seq_len * rope_factor)

    # Frequency cutoff (dimensions per YaRN)
    correction_dim = math.ceil(
        d_head // 2 * math.log(rope_factor) / math.log(rope_theta)
    )

    return {
        "original_context_length": original_seq_len,
        "extended_context_length": extended_seq_len,
        "context_extension_factor": rope_factor,
        "max_rope_cache_length": max_pos,
        "head_dimension": d_head,
        "interpolation_correction_dim": correction_dim,
        "attention_scale_correction_mscale": mscale,
        "low_freq_scaling": "extrapolate (1.0x)",
        "high_freq_scaling": f"interpolate ({rope_factor}x)",
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }


def print_rope_scaling_schedule(config: Dict[str, Any], positions: List[int]) -> None:
    """Print RoPE scaling values at various positions."""
    rope_scaling = config.get("rope_scaling", {})
    if rope_scaling.get("type") != "yarn":
        print("Not a YaRN config, skipping RoPE scaling schedule")
        return

    rope_factor = rope_scaling.get("factor", 1.0)
    original_seq_len = rope_scaling.get("original_max_position_embeddings", 2048)

    print("\nRoPE Scaling at Key Positions:")
    print("=" * 60)
    print(f"{'Position':<15} {'% of Original':<20} {'YaRN Status':<20}")
    print("-" * 60)

    for pos in positions:
        pct = (pos / original_seq_len) * 100
        if pos <= original_seq_len:
            status = "Original range (no extrapolation)"
        else:
            status = f"Extended (needs {rope_factor}x scaling)"
        print(f"{pos:<15} {pct:>6.1f}%{'':<12} {status:<20}")


def main():
    """Run validation and analysis."""
    print("=" * 70)
    print("YaRN Configuration Validation for agpt-20b-v2-512n-step-4000")
    print("=" * 70)

    # 1. Validate parameters
    print("\n1. PARAMETER VALIDATION")
    print("-" * 70)
    is_valid, message = validate_yarn_parameters(AGPT_20B_CONFIG)
    print(f"Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Message: {message}")

    if not is_valid:
        print("\n⚠ Configuration has issues. Please address before training.")
        return 1

    # 2. Calculate metrics
    print("\n2. YARN METRICS")
    print("-" * 70)
    metrics = calculate_yarn_metrics(AGPT_20B_CONFIG)
    for key, value in metrics.items():
        print(f"{key:<40} {value}")

    # 3. RoPE scaling schedule
    test_positions = [
        100, 1000, 8192, 16384, 32768, 65536, 131072
    ]
    print_rope_scaling_schedule(AGPT_20B_CONFIG, test_positions)

    # 4. Config summary
    print("\n3. CONFIGURATION SUMMARY")
    print("-" * 70)
    rope_scaling = AGPT_20B_CONFIG["rope_scaling"]
    print(f"Model: agpt-20b-v2-512n-step-4000-safetensors")
    print(f"Context Extension: {rope_scaling['original_max_position_embeddings']} → "
          f"{int(rope_scaling['original_max_position_embeddings'] * rope_scaling['factor'])} tokens")
    print(f"RoPE Theta: {AGPT_20B_CONFIG['rope_theta']}")
    print(f"Max RoPE Cache: {AGPT_20B_CONFIG['max_position_embeddings']} tokens")
    print(f"\nYaRN Parameters:")
    print(f"  - Type: {rope_scaling['type']}")
    print(f"  - Factor: {rope_scaling['factor']}x")
    print(f"  - Beta Fast: {rope_scaling['beta_fast']} (high-freq scaling)")
    print(f"  - Beta Slow: {rope_scaling['beta_slow']} (low-freq scaling)")

    # 5. Training recommendations
    print("\n4. TRAINING RECOMMENDATIONS")
    print("-" * 70)
    print(f"Suggested SEQ_LEN: {int(rope_scaling['original_max_position_embeddings'] * rope_scaling['factor'])}")
    print(f"Suggested TRAINING_STEPS: ~400 (per YaRN paper)")
    print(f"Dataset: pg19_multinews or similar long-context corpus")
    print(f"Checkpoint: agpt-20b-v2-512n-step-4000-safetensors weights")

    print("\n" + "=" * 70)
    print("✓ All validations passed. Ready for continued pretraining with YaRN.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
