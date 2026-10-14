#!/usr/bin/env python3
"""Project entry point: python main.py --config config/qwen9.yaml."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from datarsi.cli import main
if __name__ == '__main__':
    main()
