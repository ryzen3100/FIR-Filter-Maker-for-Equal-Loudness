#!/usr/bin/env python3
"""CLI for FIR filter generation with equal-loudness contours."""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

from .config import FilterConfig, PhonRangeConfig, LoggingConfig
from .services import FilterService
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
    
    # Logging options
    parser.add_argument(
        "--log", action="store_true",
        help="Enable logging to logs/ directory"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        default="INFO",
        help="Set logging level (default: INFO)"
    )

    return parser


def setup_logging(logging_config: LoggingConfig) -> logging.Logger:
    """Set up logging configuration."""
    if not logging_config.enabled:
        # Return a null logger
        logger = logging.getLogger('fir_loudness')
        logger.addHandler(logging.NullHandler())
        return logger
    
    # Create logs directory if it doesn't exist
    logging_config.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp and session ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logging_config.log_dir / f"fir_loudness_{timestamp}_{logging_config.session_id}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, logging_config.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    
    logger = logging.getLogger('fir_loudness')
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class CLIParser:
    """Handles CLI argument parsing and validation."""
    
    @staticmethod
    def parse_and_validate_args(argv=None):
        """Parse and validate CLI arguments."""
        parser = create_parser()
        return parser.parse_args(argv)
    
    @staticmethod
    def determine_curve_type(args) -> tuple[str, str]:
        """Determine curve type and ISO version from CLI args."""
        if args.fletcher:
            return "fletcher", "2023"  # Default for compatibility
        else:
            return f"iso{args.iso}", args.iso


class ConfigurationFactory:
    """Factory for creating configuration objects from CLI args."""
    
    @staticmethod
    def create_filter_config(args, curve_type: str, iso_version: str) -> FilterConfig:
        """Create FilterConfig from CLI arguments."""
        return FilterConfig.from_cli_args(
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
    
    @staticmethod
    def create_phon_config(args) -> PhonRangeConfig:
        """Create PhonRangeConfig from CLI arguments."""
        return PhonRangeConfig(
            start_phon=args.start_phon,
            end_phon=args.end_phon,
            step_phon=args.step_phon
        )
    
    @staticmethod
    def create_logging_config(args) -> LoggingConfig:
        """Create LoggingConfig from CLI arguments."""
        session_id = datetime.now().strftime("%f")
        return LoggingConfig(
            enabled=args.log,
            level=args.log_level,
            log_dir=Path("logs"),
            session_id=session_id
        )


def main(argv=None) -> int:
    """
    Main CLI entry point.
    
    Args:
        argv: Command line arguments (None for sys.argv)
        
    Returns:
        0 on success, 1 on error
    """
    try:
        # Parse arguments
        args = CLIParser.parse_and_validate_args(argv)
        
        # Create configurations
        curve_type, iso_version = CLIParser.determine_curve_type(args)
        filter_config = ConfigurationFactory.create_filter_config(args, curve_type, iso_version)
        phon_config = ConfigurationFactory.create_phon_config(args)
        logging_config = ConfigurationFactory.create_logging_config(args)
        
        # Set up logging
        logger = setup_logging(logging_config)
        logger.info("Starting FIR filter generation")
        logger.info(f"Configuration: {filter_config.to_dict()}")
        logger.info(f"Phon range: {phon_config.start_phon}-{phon_config.end_phon} phon")
        
        # Execute generation
        service = FilterService(filter_config, phon_config, logger)
        count = service.execute()
        
        logger.info(f"Generated {count} filter(s) into: {filter_config.output_dir}")
        print(f"Generated {count} filter(s) into: {filter_config.output_dir}")
        return 0
        
    except ValidationError as e:
        if 'logger' in locals():
            logger.error(f"Validation error: {e}")
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
