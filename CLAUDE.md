# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIR-Filter-Maker-for-Equal-Loudness generates Finite Impulse Response (FIR) filters based on ISO 226 equal-loudness contours. The filters maintain consistent tonal balance at lower volume levels (60-80 dB) and can be used in Equalizer APO or other convolution hosts.

## Architecture

- **Entry Point**: `fir_loudness_cli.py` - thin wrapper that delegates to CLI module
- **CLI**: `src/cli.py` - argument parsing and orchestration
- **Core Modules**:
  - `src/iso_data.py` - ISO 2003/2023 equal-loudness contour data
  - `src/interpolation.py` - fine interpolation between ISO curves
  - `src/design.py` - FIR filter design using scipy.signal.firwin2
  - `src/io_utils.py` - WAV file output and CSV export utilities

## Key Commands

```bash
# Generate a single filter (60-80 phon transition)
python fir_loudness_cli.py --start-phon 60 --end-phon 80

# Custom sampling rate and filter length
python fir_loudness_cli.py --start-phon 50 --end-phon 75 --fs 44100 --taps 8192

# Stereo output with CSV export
python fir_loudness_cli.py --start-phon 65 --end-phon 85 --channels 2 --export-csv

# Batch generation with fine control
python fir_loudness_cli.py --start-phon 40 --end-phon 90 --step-phon 5 --out-dir batch_output
```

## Parameters Reference

- `--iso`: Select dataset (2003|2023, default: 2023)
- `--fs`: Sampling rate Hz (default: 48000)
- `--taps`: Filter length/taps (default: 65536)
- `--start-phon`: Beginning phon level (required)
- `--end-phon`: Target phon level (required) 
- `--step-phon`: Step size for multiple filters (default: 0.1)
- `--channels`: 1=mono, 2=stereo (default: 1)
- `--format`: wav sample format (float32|pcm16, default: float32)

## File Generation

Output files follow pattern: `ISO{year}_fs{fs}_t{taps}_{start_phon}-{end_phon}_filter.wav`

Optional exports:
- `--export-csv`: Target response grid (CSV)
- `--export-fir-resp`: Designed FIR magnitude response (CSV)

## Dependencies

```bash
pip install numpy scipy
```

The code uses standard library for `wave` module and scipy for filter design.