#!/usr/bin/env python3
# Thin entry point that delegates to cli.main
import sys
from src.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
