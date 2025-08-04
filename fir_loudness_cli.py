#!/usr/bin/env python3
# Thin entry point that delegates to cli.main
import sys
import os

# Ensure the src package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
