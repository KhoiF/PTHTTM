#!/usr/bin/env python3
"""Assignment 03 diabetes_large CLI and localhost deploy app."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy_support import run

if __name__ == "__main__":
    raise SystemExit(run("diabetes_large", 8004))
