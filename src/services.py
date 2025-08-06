"""Service classes for decoupled business logic."""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import logging

from .config import FilterConfig, PhonRangeConfig
from .repositories import CurveRepositoryFactory
from .interpolation import create_fine_interpolated_curves, round_phon_key
from .design import (design_fir_filter_from_phon_levels,
                     compute_relative_gains, prepare_target_response)
from .io_utils import save_filter_to_wav, save_response_csv
from scipy import signal


class FilterGenerationService:
    """Service for FIR filter generation operations."""

    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.phon_config = phon_config
        self.logger = logger or logging.getLogger('fir_loudness')
        self._repository = None
        self._fine_curves: Optional[Dict[float, list]] = None
        self._freq_points: Optional[np.ndarray] = None

    def _get_repository(self):
        """Get the appropriate curve repository."""
        if self._repository is None:
            self._repository = CurveRepositoryFactory.create_repository(
                self.config.curve_type,
                self.config.iso_version
            )
        return self._repository

    def _ensure_curves_initialized(self) -> None:
        """Initialize curves and frequency points if not already."""
        if self._fine_curves is None:
            repo = self._get_repository()
            curves = repo.get_curves()
            freqs = repo.get_frequencies()
            s = round_phon_key(self.phon_config.start_phon)
            e = round_phon_key(self.phon_config.end_phon)
            
            # Convert to lists for compatibility
            curves_as_lists = {k: [float(x) for x in v] for k, v in curves.items()}
            freqs_list = [float(x) for x in freqs]
            
            self._fine_curves = create_fine_interpolated_curves(
                curves_as_lists, freqs_list, step=0.1, needed_range=(s, e)
            )
            self._freq_points = freqs
            self.logger.debug(f"Initialized {repo.get_curve_type()} curves "
                             f"for range {s}-{e} phon")

    @property
    def fine_curves(self) -> Dict[float, list]:
        self._ensure_curves_initialized()
        assert self._fine_curves is not None
        return self._fine_curves

    @property
    def freq_points(self) -> np.ndarray:
        self._ensure_curves_initialized()
        assert self._freq_points is not None
        return self._freq_points

    def generate_single_filter(self, source_phon: float,
                               target_phon: float) -> np.ndarray:
        """Generate a single FIR filter."""
        self._ensure_curves_initialized()


        curves: Dict[float, list] = self.fine_curves
        freqs: np.ndarray = self.freq_points
        fir = design_fir_filter_from_phon_levels(
            source_phon, target_phon, curves,
            freqs.tolist(), self.config
        )

        self.logger.debug(f"Generated FIR filter with {len(fir)} taps for "
                         f"{source_phon:.1f}→{target_phon:.1f} phon")
        return fir

    def estimate_total_filters(self) -> int:
        """Estimate total number of filters to be generated."""
        return self.phon_config.estimate_total_filters()

    def generate_phon_levels(self):
        """Generate phon levels based on configuration."""
        return self.phon_config.generate_phon_levels()


class FileNamingService:
    """Service for generating consistent file names."""

    def __init__(self, config: FilterConfig):
        self.config = config

    def generate_filter_filename(self, source_phon: float,
                                target_phon: float) -> str:
        """Generate consistent file name for filter."""
        curve_identifier = self._get_curve_identifier()
        return (f"{curve_identifier}_fs{self.config.fs}_t{self.config.numtaps}_"
                f"{source_phon:.1f}-{target_phon:.1f}_filter.wav")

    def generate_response_filename(self, source_phon: float,
                                  target_phon: float) -> str:
        """Generate file name for response CSV."""
        base_name = self.generate_filter_filename(source_phon, target_phon)
        return base_name.replace("_filter.wav", "_response.csv")

    def generate_fir_response_filename(self, source_phon: float,
                                       target_phon: float) -> str:
        """Generate file name for FIR magnitude response CSV."""
        base_name = self.generate_filter_filename(source_phon, target_phon)
        return base_name.replace("_filter.wav", "_fir_mag.csv")

    CURVE_MAP = {
        "fletcher": "FLETCHER",
        "iso2003": "ISO2003",
        "iso2023": "ISO2023"
    }

    def _get_curve_identifier(self) -> str:
        """Get curve type identifier for file naming."""
        return self.CURVE_MAP.get(self.config.curve_type, "UNKNOWN")


class ExportService:
    """Service for handling file exports."""

    def __init__(self, config: FilterConfig,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger('fir_loudness')

    def save_filter(self, fir: np.ndarray, filename: str) -> Path:
        """Save FIR filter to WAV file."""
        file_path = self.config.output_dir / filename
        save_filter_to_wav(
            fir, str(file_path), self.config.fs,
            channels=self.config.channels,
            sample_format=self.config.sample_format
        )
        self.logger.debug(f"Saved filter to: {file_path}")
        return file_path

    def export_target_response(self, source_phon: float, target_phon: float,
                               fine_curves: Dict[float, list],
                               freq_points: np.ndarray,
                               filename: str) -> None:
        """Export target response grid to CSV."""
        if not self.config.export_csv:
            return

        relative_gains_db = compute_relative_gains(
            fine_curves, source_phon, target_phon, freq_points.tolist(),
            self.config.use_smoothing, self.config.smooth_window
        )

        freqs, gains = prepare_target_response(relative_gains_db,
                                              freq_points.tolist(),
                                              self.config)

        file_path = self.config.output_dir / filename
        save_response_csv(freqs, gains, str(file_path))
        self.logger.debug("Exported target response CSV")

    def export_fir_response(self, fir: np.ndarray, filename: str) -> None:
        """Export actual FIR magnitude response to CSV."""
        if not self.config.export_fir_response:
            return


        w, h = signal.freqz(fir, worN=4096, fs=self.config.fs)
        mag = np.abs(h)

        # Ensure arrays are numpy arrays for type compatibility
        w_array = np.asarray(w)
        mag_array = np.asarray(mag)

        file_path = self.config.output_dir / filename
        save_response_csv(w_array, mag_array, str(file_path))
        self.logger.debug("Exported FIR magnitude response CSV")


class FilterService:
    """High-level service for filter generation workflow."""

    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.phon_config = phon_config
        self.logger = logger or logging.getLogger('fir_loudness')

        self.generation_service = FilterGenerationService(config, phon_config,
                                                         logger)
        self.naming_service = FileNamingService(config)
        self.export_service = ExportService(config, logger)

    def execute(self) -> int:
        """Execute the complete filter generation workflow."""
        self.logger.info("Starting filter generation workflow")

        total_filters = self.generation_service.estimate_total_filters()
        self.logger.info(f"Estimated total filters: {total_filters}")

        count = 0
        for source_phon in self.generation_service.generate_phon_levels():
            self._process_single_filter(source_phon,
                                        self.phon_config.end_phon)
            count += 1
            self.logger.debug(f"Completed filter {count}/{total_filters}")

        self.logger.info(f"Filter generation completed. Generated {count} "
                         "filters")
        return count

    def _process_single_filter(self, source_phon: float,
                               target_phon: float) -> None:
        """Process a single filter generation."""
        self.logger.info(f"Generating filter: {source_phon:.1f} → "
                         f"{target_phon:.1f} phon")

        # Generate filter
        fir = self.generation_service.generate_single_filter(source_phon,
                                                            target_phon)

        # Generate file names
        filter_filename = self.naming_service.generate_filter_filename(
            source_phon, target_phon)
        response_filename = self.naming_service.generate_response_filename(
            source_phon, target_phon)
        fir_response_filename = self.naming_service.generate_fir_response_filename(
            source_phon, target_phon)

        # Save and export
        self.export_service.save_filter(fir, filter_filename)

        # Export responses if requested
        if (self.generation_service._fine_curves is None or
                self.generation_service._freq_points is None):
            self.logger.warning("Cannot export target response: curves not "
                                "initialized")
            return

        self.export_service.export_target_response(
            source_phon, target_phon,
            self.generation_service._fine_curves,
            self.generation_service._freq_points,
            response_filename
        )
        self.export_service.export_fir_response(fir, fir_response_filename)
