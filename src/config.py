"""Configuration classes for the FIR filter maker."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .validation import (
    validate_sampling_rate, validate_filter_taps, validate_phon_level,
    validate_step_size, validate_channels, validate_sample_format,
    validate_dc_gain_mode, validate_nyq_gain_db, validate_iso_version,
    validate_curve_type, validate_grid_points, validate_directory_path
)


@dataclass
class FilterConfig:
    """Configuration for FIR filter design."""
    
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
    
    def __post_init__(self):
        """Validate all configuration parameters."""
        self.fs = validate_sampling_rate(self.fs)
        self.numtaps = validate_filter_taps(self.numtaps)
        self.iso_version = validate_iso_version(self.iso_version)
        self.curve_type = validate_curve_type(self.curve_type)
        self.channels = validate_channels(self.channels)
        self.sample_format = validate_sample_format(self.sample_format)
        self.dc_gain_mode = validate_dc_gain_mode(self.dc_gain_mode)
        self.nyq_gain_db = validate_nyq_gain_db(self.nyq_gain_db)
        self.grid_points = validate_grid_points(self.grid_points)
        self.output_dir = validate_directory_path(self.output_dir)
        
        if self.smooth_window < 1 or self.smooth_window > 15:
            raise ValueError("Smoothing window must be between 1 and 15")
        if self.smooth_window % 2 == 0:
            raise ValueError("Smoothing window must be odd")
    
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
class PhonRangeConfig:
    """Configuration for phon level ranges."""
    
    start_phon: float
    end_phon: float
    step_phon: float
    
    def __post_init__(self):
        """Validate phon range configuration."""
        self.start_phon = validate_phon_level(self.start_phon, "Start phon level")
        self.end_phon = validate_phon_level(self.end_phon, "End phon level")
        self.step_phon = validate_step_size(self.step_phon)
        
        if self.start_phon > self.end_phon:
            raise ValueError(
                f"Start phon level ({self.start_phon}) must be <= end phon level ({self.end_phon})"
            )
    
    def generate_phon_levels(self):
        """Generate phon levels based on range configuration."""
        from .interpolation import round_phon_key
        
        start = round_phon_key(self.start_phon)
        end = round_phon_key(self.end_phon)
        
        n_steps = int(round((end - start) / self.step_phon))
        for i in range(n_steps + 1):
            yield start + i * self.step_phon
    
    def estimate_total_filters(self) -> int:
        """Estimate total number of filters to be generated."""
        from .interpolation import round_phon_key
        
        start = round_phon_key(self.start_phon)
        end = round_phon_key(self.end_phon)
        n_steps = int(round((end - start) / self.step_phon))
        return n_steps + 1