# FIR-Filter-Maker-for-Equal-Loudness
Generating FIR filters to balance tones at low volumes using ISO 226 equal-loudness contours.

## Overview
A modern Python CLI tool that generates Finite Impulse Response (FIR) filters to maintain consistent tonal balance across different volume levels. Uses ISO 226:2003/2023 equal-loudness contours to create EQ adjustments that compensate for human hearing characteristics at lower volumes.

Perfect for use with Equalizer APO, APO-loudness, Easy Convolver, or other convolution-based EQ systems.

## Features
- **Multiple Standards**: Choose between ISO 226:2003/2023 or Fletcher-Munson equal-loudness contours
- **Flexible Range Generation**: Generate filters for any phon level range
- **Multiple Formats**: WAV output in float32 or int16 PCM formats
- **Stereo Support**: Mono or stereo channel output
- **Advanced Tuning**: Configurable smoothing, DC gain, and Nyquist response
- **Export Options**: Filter response CSV export available
- **Performance Benchmarking**: Built-in latency analysis for real-time applications
- **Production Ready**: Comprehensive input validation and security

## Quick Start

### Installation
```bash
# Clone this repository
git clone https://github.com/ryzen3100/FIR-Filter-Maker-for-Equal-Loudness.git
cd FIR-Filter-Maker-for-Equal-Loudness

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy

# Optional: Install development tools
pip install pyright  # For type checking
```

### Basic Usage
```bash
# Generate single filter (40→50 phon transition)
python fir_loudness_cli.py --start-phon 40 --end-phon 50

# Use Fletcher-Munson curves instead of ISO
python fir_loudness_cli.py --start-phon 40 --end-phon 50 --fletcher

# Custom sampling rate and filter length
python fir_loudness_cli.py --start-phon 60 --end-phon 80 --fs 44100 --taps 8192

# Stereo output with fine resolution
python fir_loudness_cli.py --start-phon 35 --end-phon 85 --step-phon 0.5 --channels 2

# Batch generation with CSV export
python fir_loudness_cli.py --start-phon 50 --end-phon 90 --step-phon 2.5 --export-csv

# Enable logging for debugging
python fir_loudness_cli.py --start-phon 40 --end-phon 50 --log --log-level DEBUG
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start-phon` | Source phon level (0-100) | **Required** |
| `--end-phon` | Target phon level (0-100) | **Required** |
| `--step-phon` | Step size for start phon (0, 10] | 0.1 |
| `--fs` | Sampling rate (Hz) | 48000 |
| `--taps` | Filter length (4-262144) | 65536 |
| `--channels` | Output channels (1=mono, 2=stereo) | 1 |
| `--format` | Sample format (float32/pcm16) | float32 |
| `--iso` | ISO standard (2003/2023) | 2023 |
| `--fletcher` | Use Fletcher-Munson contours (mutually exclusive with --iso) | False |
| `--out-dir` | Output directory | output |
| `--log` | Enable logging to logs/ directory | False |
| `--log-level` | Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | INFO |

### Advanced Options
- `--smooth-db`: Enable smoothing across ISO points
- `--smooth-window`: Smoothing window size (odd integer)
- `--dc-gain-mode`: DC gain control (first_iso/unity)
- `--nyq-gain-db`: Nyquist gain control (dB)
- `--grid-points`: Frequency grid resolution
- `--export-csv`: Export response data as CSV
- `--export-fir-resp`: Export actual FIR response data

### Logging Options
- `--log`: Enable logging to logs/ directory
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Performance Analysis
- `--benchmark`: Run latency benchmark to determine optimal tap sizes for your system

## Examples

### Simple Use Case
Generate a filter to compensate when listening at 70dB while the content was mastered at 85dB:
```bash
python fir_loudness_cli.py --start-phon 70 --end-phon 85
```

### Batch Generation
Create filters for every 2.5 phon step from 45 to 85:
```bash
python fir_loudness_cli.py --start-phon 45 --end-phon 85 --step-phon 2.5
```

### Performance Benchmarking
Determine optimal tap sizes for your system:
```bash
python fir_loudness_cli.py --benchmark
```

### Custom Parameters
Generate with specific technical requirements:
```bash
python fir_loudness_cli.py \
  --start-phon 60 \
  --end-phon 75 \
  --fs 44100 \
  --taps 16384 \
  --channels 2 \
  --format pcm16 \
  --out-dir my_filters
```

## Dependencies
```bash
pip install numpy scipy

# For development (optional)
pip install pyright  # Type checking
```

## Output
Filter files are named with metadata:
```
ISO2023_fs48000_t65536_60.0-75.0_filter.wav
FLETCHER_fs48000_t1024_40.0-60.0_filter.wav
```
Where:
- `ISO2023`/`FLETCHER`: Curve standard used
- `fs48000`: Sampling rate (48000 Hz)
- `t65536`: Filter length (65536 taps)
- `60.0-75.0`: Source → Target phon levels

## Security & Validation
All parameters are validated with:
- File path sanitization preventing directory traversal
- Numeric bounds checking (phon 0-100, sampling rate 8k-192kHz)
- Input type and range validation
- Zero division protection and NaN handling
- Type safety with mypy and pyright compatibility

## Usage in Equalizer Applications

### Equalizer APO
1. Copy generated WAV files to Equalizer APO config directory
2. Reference filters by phon level differences
3. Use with smart gain switching or manual control

### APO-Loudness
Directly compatible with APO-Loudness convolution setup.

## Project Structure
```
src/
├── cli.py               # Main CLI interface
├── business.py          # Core filtration logic
├── config.py            # Configuration classes
├── design.py            # FIR filter design
├── interpolation.py     # Curve interpolation
├── io_utils.py          # File I/O with pathlib
├── iso_data.py          # ISO contour data
├── fletcher_data.py     # Fletcher-Munson contour data
├── benchmark.py         # Performance latency analysis
└── validation.py        # Input validation & security

logs/                    # Log files (when logging enabled)
output/                  # Generated filter files
```

## Legacy Information
The original deprecated scripts (`FIR_LOUDNESS.py`, `FIR_LOUDNESS_2023.py`) have been replaced with the modern `fir_loudness_cli.py` interface. All functionality is preserved but with improved CLI, security, and performance.

## Original Author & Repository
This project was originally created by **grisys83** and is based on the original repository at:  
**https://github.com/grisys83/FIR-Filter-Maker-for-Equal-Loudness**

This repository contains refactored and enhanced versions of the original code, maintaining the same core functionality while adding security, performance, and architecture improvements.

## License
GNU General Public License version 3 (GPLv3)

## Data Sources
- ISO 226:2003/2023 equal-loudness contours
- Fletcher-Munson equal-loudness contours (classic 1933 data)
- Missing 20000 Hz and 16000 Hz frequencies in 2023 standard preserved from 2003

## Support
For questions about the original project: 136304138+grisys83@users.noreply.github.com