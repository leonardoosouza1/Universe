#!/usr/bin/env python3
"""Validate champion DNA: check if it's being saved/loaded between runs."""

import os
import subprocess
import json
from pathlib import Path
import struct
import time

def read_dna_info(dna_path):
    """Extract DNA metadata without full parsing."""
    if not os.path.exists(dna_path):
        return None

    try:
        with open(dna_path, 'rb') as f:
            data = f.read()

        if len(data) < 512:
            return {"size": len(data), "valid": False, "reason": "too small"}

        # Check header
        if not data.startswith(b"UNIV"):
            return {"size": len(data), "valid": False, "reason": "bad header"}

        version = data[4]
        if version > 7:
            return {"size": len(data), "valid": False, "reason": f"unknown version {version}"}

        # Extract some fields (rough positions, depends on version)
        try:
            # Offset 256: fast_weights[192]
            fast_sum = sum(data[256:256+192])
            # Offset 448: fast_scale (f32)
            fast_scale = struct.unpack('<f', data[448:452])[0]

            return {
                "size": len(data),
                "valid": True,
                "version": version,
                "fast_scale": fast_scale,
                "fast_sum": fast_sum,
            }
        except:
            return {"size": len(data), "valid": True, "version": version, "note": "partial parse"}

    except Exception as e:
        return {"size": 0, "valid": False, "reason": str(e)}

def main():
    print("\n🧬 DNA VALIDATION REPORT\n")

    # Check current CHAMPION.dna
    champ_info = read_dna_info("CHAMPION.dna")
    print("📁 Current CHAMPION.dna:")
    if champ_info:
        print(f"  Size: {champ_info.get('size')} bytes")
        print(f"  Valid: {champ_info.get('valid', False)}")
        if champ_info.get('version'):
            print(f"  Version: {champ_info['version']}")
        if champ_info.get('fast_scale'):
            print(f"  Fast scale: {champ_info['fast_scale']:.4f}")
        if champ_info.get('fast_sum'):
            print(f"  Fast weights sum: {champ_info['fast_sum']}")
    else:
        print("  ❌ Not found or unreadable")

    print()

    # Check telemetry for DNA saving patterns
    telemetry_dir = Path("telemetry")
    if telemetry_dir.exists():
        runs = list(telemetry_dir.iterdir())
        print(f"📊 Telemetry runs: {len(runs)} found")

        for run_dir in sorted(runs)[:5]:  # Check first 5
            meta_file = run_dir / "run_meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                print(f"\n  Run: {run_dir.name}")
                print(f"    Experiment: {meta.get('experiment_name', 'n/a')}")

                # Check if DNA file exists
                gen_file = run_dir / "generations.ndjson"
                if gen_file.exists():
                    with open(gen_file) as f:
                        lines = f.readlines()
                    print(f"    Generations: {len(lines)}")

    print("\n✅ DNA VALIDATION COMPLETE")
    print("Note: DNA structure depends on version. Check universe_config.json for DNA_VERSION.")

if __name__ == "__main__":
    main()
