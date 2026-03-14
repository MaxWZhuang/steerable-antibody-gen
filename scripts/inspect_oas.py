#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.data.oas import read_oas_table

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect one raw OAS file.")
    p.add_argument("path", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metadata, df = read_oas_table(args.path)
    print("=== METADATA ===")
    print(json.dumps(metadata, indent=2))
    print("=== COLUMNS ===")
    print(df.columns.tolist())
    print("=== HEAD ===")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
