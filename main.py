#!/usr/bin/env python3
"""YOLO-LAB-CLI entry point."""

import sys
import os

# Ensure scripts/ is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from train_segment import main

if __name__ == "__main__":
    main()
