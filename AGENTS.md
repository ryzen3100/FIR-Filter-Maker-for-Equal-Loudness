# AGENTS.md - FIR Filter Maker

## Build/Test Commands
```bash
# Quick validation test
python fir_loudness_cli.py --start-phon 40 --end-phon 50 --fs 48000 --taps 1024

# Check imports & CLI
python -c "from src.cli import create_parser; p=create_parser(); p.parse_args(['--help'])"

# Validate all modules
python -c "import src.cli, src.business, src.config, src.validation"
```

## Code Style
- **Imports**: Absolute only (`from src.config import FilterConfig`)
- **Types**: Dataclasses for config, Optional for nullable args
- **Naming**: snake_case functions/vars, PascalCase classes
- **Error Handling**: Custom ValidationError, try/catch in CLI, return int codes
- **Paths**: pathlib.Path with validate_directory_path() security
- **Validation**: __post_init__ via validation.py, security-focused
- **Formatting**: PEP 8, 4-space indent, 100 char limit