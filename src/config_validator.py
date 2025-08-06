"""
Configuration validator module.
Separates validation responsibilities from configuration objects.
"""

from typing import Union, Optional
from pathlib import Path

from .validation import (
    validate_sampling_rate, 
    validate_filter_taps, 
    validate_phon_level,
    validate_step_size, 
    validate_sample_format,
    validate_dc_gain_mode, 
    validate_nyq_gain_db, 
    validate_iso_version,
    validate_curve_type, 
    validate_grid_points, 
    validate_directory_path
)
from .validation import ChoiceValidator
from .config import FilterConfig, PhonRangeConfig, LoggingConfig


class ConfigValidator:
    """Handles validation for configuration objects."""
    
    @staticmethod
    def validate_filter_config(config: FilterConfig) -> None:
        """Validate all parameters in FilterConfig."""
        config.fs = validate_sampling_rate(config.fs)
        config.numtaps = validate_filter_taps(config.numtaps)
        config.iso_version = validate_iso_version(config.iso_version)
        config.curve_type = validate_curve_type(config.curve_type)
        config.channels = ChoiceValidator([1, 2]).validate(config.channels, "Channels")
        config.sample_format = validate_sample_format(config.sample_format)
        config.dc_gain_mode = validate_dc_gain_mode(config.dc_gain_mode)
        config.nyq_gain_db = validate_nyq_gain_db(config.nyq_gain_db)
        config.grid_points = validate_grid_points(config.grid_points)
        config.output_dir = validate_directory_path(config.output_dir)

        if config.smooth_window < 1 or config.smooth_window > 15:
            raise ValueError("Smoothing window must be between 1 and 15")
        if config.smooth_window % 2 == 0:
            raise ValueError("Smoothing window must be odd")
    
    @staticmethod
    def validate_phon_range_config(config: PhonRangeConfig) -> None:
        """Validate phon range configuration."""
        config.start_phon = validate_phon_level(config.start_phon, "Start phon level")
        config.end_phon = validate_phon_level(config.end_phon, "End phon level")
        config.step_phon = validate_step_size(config.step_phon)

        if config.start_phon > config.end_phon:
            # Swap start and end instead of failing for better UX
            config.start_phon, config.end_phon = config.end_phon, config.start_phon
    
    @staticmethod
    def validate_logging_config(config: LoggingConfig) -> None:
        """Validate logging configuration."""
        if config.enabled:
            config.log_dir = validate_directory_path(config.log_dir)
            if config.level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ValueError(f"Invalid log level: {config.level}")
    
    @classmethod
    def validate_all(cls, filter_config: FilterConfig, 
                    phon_config: Optional[PhonRangeConfig] = None,
                    logging_config: Optional[LoggingConfig] = None) -> None:
        """Validate all provided configurations."""
        cls.validate_filter_config(filter_config)
        if phon_config:
            cls.validate_phon_range_config(phon_config)
        if logging_config:
            cls.validate_logging_config(logging_config)