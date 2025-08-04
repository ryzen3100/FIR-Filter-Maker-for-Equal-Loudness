#!/usr/bin/env python3
"""CLI for FIR filter generation with equal-loudness contours."""

import argparse
import sys
from pathlib import Path

from .config import FilterConfig, PhonRangeConfig
from .business import FilterGenerator
from .validation import ValidationError


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="FIR filter generator for equal-loudness transitions (ISO226)"
    )
    
    # Curve selection (mutually exclusive)
    curve_group = parser.add_mutually_exclusive_group()
    curve_group.add_argument(
        "--iso", choices=["2003", "2023"], default="2023",
        help="Select ISO dataset (default: 2023)"
    )
    curve_group.add_argument(
        "--fletcher", action="store_true",
        help="Use Fletcher-Munson equal-loudness contours instead of ISO"
    )
    parser.add_argument(
        "--fs", type=int, default=48000,
        help="Sampling rate in Hz (default: 48000)"
    )
    parser.add_argument(
        "--taps", type=int, default=65536,
        help="Number of FIR taps (default: 65536)"
    )
    parser.add_argument(
        "--start-phon", type=float, required=True,
        help="Start phon level (e.g., 60.0)"
    )
    parser.add_argument(
        "--end-phon", type=float, required=True,
        help="End phon level (e.g., 85.0)"
    )
    parser.add_argument(
        "--step-phon", type=float, default=0.1,
        help="Step for start phon (default: 0.1)"
    )
    parser.add_argument(
        "--channels", type=int, choices=[1, 2], default=1,
        help="Number of channels in WAV (1=mono, 2=stereo, default: 1)"
    )
    parser.add_argument(
        "--format", dest="sample_format", choices=["float32", "pcm16"], default="float32",
        help="Output WAV sample format (default: float32)"
    )
    parser.add_argument(
        "--out-dir", default="output",
        help="Output directory (default: output)"
    )

    # Advanced parameters
    parser.add_argument(
        "--smooth-db", action="store_true",
        help="Enable mild smoothing in dB across ISO points"
    )
    parser.add_argument(
        "--smooth-window", type=int, default=3,
        help="Smoothing window length (odd integer, default: 3)"
    )
    parser.add_argument(
        "--dc-gain-mode", choices=["first_iso", "unity"], default="first_iso",
        help="DC gain choice: extend first ISO gain or force unity (default: first_iso)"
    )
    parser.add_argument(
        "--nyq-gain-db", type=float, default=None,
        help="Nyquist gain in dB. Default None tapers to 0 linear"
    )
    parser.add_argument(
        "--grid-points", type=int, default=2048,
        help="Number of frequency grid points for target response (default: 2048)"
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export the PCHIP target grid (freqs,gains) for each filter"
    )
    parser.add_argument(
        "--export-fir-resp", action="store_true",
        help="Export the designed FIR magnitude response as CSV"
    )

    return parser


def main(argv=None) -> int:
    """
    Main CLI entry point.
    
    Args:
        argv: Command line arguments (None for sys.argv)
        
    Returns:
        0 on success, 1 on error
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        # Determine curve type
        if args.fletcher:
            curve_type = "fletcher"
            iso_version = "2023"  # Default for compatibility
        else:
            curve_type = f"iso{args.iso}"
            iso_version = args.iso

        # Create configuration objects
        filter_config = FilterConfig.from_cli_args(
            fs=args.fs,
            numtaps=args.taps,
            iso=iso_version,
            curve_type=curve_type,
            channels=args.channels,
            sample_format=args.sample_format,
            dc_gain_mode=args.dc_gain_mode,
            nyq_gain_db=args.nyq_gain_db,
            grid_points=args.grid_points,
            smooth_db=args.smooth_db,
            smooth_window=args.smooth_window,
            export_csv=args.export_csv,
            export_fir_resp=args.export_fir_resp,
            out_dir=Path(args.out_dir)
        )
        
        phon_config = PhonRangeConfig(
            start_phon=args.start_phon,
            end_phon=args.end_phon,
            step_phon=args.step_phon
        )
        
        # Execute generation
        generator = FilterGenerator(filter_config, phon_config)
        count = generator.execute()
        
        print(f"Generated {count} filter(s) into: {filter_config.output_dir}")
        return 0
        
    except ValidationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
