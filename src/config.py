"""Pure DTO configuration classes for the FIR filter maker.
Zero validation dependencies - validation handled separately."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class FilterConfig:
    """Pure data transfer object for FIR filter design configuration."""

    fs: int
    numtaps: int
    iso_version: str
    curve_type: str
    channels: int
    sample_format: str
    dc_gain_mode: str
    nyq_gain_db: Optional[float]
    grid_points: int
    use_smoothing: bool
    smooth_window: int
    export_csv: bool
    export_fir_response: bool
    output_dir: Path

    @classmethod
    def from_cli_args(
        cls,
        fs: int,
        numtaps: int,
        iso: str,
        curve_type: str,
        channels: int,
        sample_format: str,
        dc_gain_mode: str,
        nyq_gain_db: Optional[float],
        grid_points: int,
        smooth_db: bool,
        smooth_window: int,
        export_csv: bool,
        export_fir_resp: bool,
        out_dir: Path
    ) -> "FilterConfig":
        """Create configuration from CLI arguments."""
        return cls(
            fs=fs,
            numtaps=numtaps,
            iso_version=iso,
            curve_type=curve_type,
            channels=channels,
            sample_format=sample_format,
            dc_gain_mode=dc_gain_mode,
            nyq_gain_db=nyq_gain_db,
            grid_points=grid_points,
            use_smoothing=smooth_db,
            smooth_window=smooth_window,
            export_csv=export_csv,
            export_fir_response=export_fir_resp,
            output_dir=out_dir
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for debugging."""
        return {
            'fs': self.fs,
            'numtaps': self.numtaps,
            'iso_version': self.iso_version,
            'curve_type': self.curve_type,
            'channels': self.channels,
            'sample_format': self.sample_format,
            'dc_gain_mode': self.dc_gain_mode,
            'nyq_gain_db': self.nyq_gain_db,
            'grid_points': self.grid_points,
            'use_smoothing': self.use_smoothing,
            'smooth_window': self.smooth_window,
            'export_csv': self.export_csv,
            'export_fir_response': self.export_fir_response,
            'output_dir': str(self.output_dir)
        }


@dataclass
class LoggingConfig:
    """Pure data transfer object for logging configuration."""

    enabled: bool
    level: str
    log_dir: Path
    session_id: str


@dataclass
class PhonRangeConfig:
    """Pure data transfer object for phon level range configuration."""

    start_phon: float
    end_phon: float
    step_phon: float
