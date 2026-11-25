#!/usr/bin/env python3
"""Benchmark runner (`benchmarks/run_benchmarks.py`) regenerating `benchmarks/latest_results.json`."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from veri_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
