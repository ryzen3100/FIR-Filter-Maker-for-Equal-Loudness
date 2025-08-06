"""Refactored services using proper domain layer architecture.
Implements clean separation between orchestration and domain logic."""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import logging

from .config import FilterConfig, PhonRangeConfig
from .phon_service import PhonService
from .repositories import CurveRepositoryFactory
from .design import design_fir_filter_from_phon_levels
from .io_utils import save_filter_to_wav, save_response_csv
from .domain.equal_loudness_curves import EqualLoudnessCurves
from .domain.fir_design_spec import FIRDesignSpec
from .domain.phon_transition import PhonTransition
from scipy import signal


class LegacyFilterGenerator:
    """
    Legacy adapter for backward compatibility during transition.
    Provides old interface while internally using new architecture.
    """
    
    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig, logger=None):
        self._service = FilterService(config, phon_config, logger)
        
    def execute(self) -> int:
        """Legacy method for backward compatibility."""
        return self._service.execute()


class FilterGenerationService:
    """Refactored Service using domain layer for curve handling."""

    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.phon_config = phon_config
        self.logger = logger or logging.getLogger('fir_loudness')
        self._repository = None
        self._equal_loudness_curves: Optional[EqualLoudnessCurves] = None
        self._phon_service = PhonService()

    def _get_repository(self):
        """Get the appropriate curve repository."""
        if self._repository is None:
            self._repository = CurveRepositoryFactory.create_repository(
                self.config.curve_type,
                self.config.iso_version
            )
        return self._repository

    def _ensure_curves_initialized(self) -> None:
        """Initialize domain curves if not already."""
        if self._equal_loudness_curves is None:
            repo = self._get_repository()
            curves = repo.get_curves()
            frequencies = repo.get_frequencies()
            
            # Convert to Python list format for domain model
            curves_dict = {float(k): v.tolist() for k, v in curves.items()}
            frequencies_list = frequencies.tolist()
            
            self._equal_loudness_curves = EqualLoudnessCurves(
                curves_dict, frequencies_list
            )
            
            # Initialize for the phon range
            self._equal_loudness_curves.initialize_fine_curves(
                self.phon_config.start_phon,
                self.phon_config.end_phon,
                step=0.1
            )
            
            self.logger.debug(
                f"Initialized {repo.get_curve_type()} curves for range "
                f"{self.phon_config.start_phon}-{self.phon_config.end_phon} phon"
            )

    @property
    def equal_loudness_curves(self) -> EqualLoudnessCurves:
        self._ensure_curves_initialized()
        assert self._equal_loudness_curves is not None
        return self._equal_loudness_curves

    def generate_single_filter(self, source_phon: float,
                               target_phon: float) -> np.ndarray:
        """Generate a single FIR filter using domain layer."""
        curves = self.equal_loudness_curves
        
        # Create design specification using domain model
        transition = PhonTransition(source_phon, target_phon)
        
        # Compute relative gains using domain layer
        relative_gains = curves.compute_relative_gains(
            source_phon, target_phon,
            use_smoothing=self.config.use_smoothing,
            smooth_window=self.config.smooth_window
        )
        
        # Prepare target response using existing design utilities
        from .design import prepare_target_response
        freqs, gains = prepare_target_response(
            relative_gains, curves.get_frequencies(), self.config
        )
        
        # Design FIR filter using the prepared inputs
        fir = design_fir_filter_from_phon_levels(
            source_phon, target_phon,
            curves._fine_curves if curves._fine_curves else {},  # For compatibility
            curves.get_frequencies(),
            self.config
        )

        self.logger.debug(f"Generated FIR filter with {len(fir)} taps for "
                         f"{source_phon:.1f}→{target_phon:.1f} phon")
        return fir

    def estimate_total_filters(self) -> int:
        """Estimate total number of filters to be generated."""
        return self._phon_service.estimate_total_filters(
            self.phon_config.start_phon,
            self.phon_config.end_phon,
            self.phon_config.step_phon
        )

    def generate_phon_levels(self):
        """Generate phon levels based on configuration."""
        return self._phon_service.generate_phon_levels(
            self.phon_config.start_phon,
            self.phon_config.end_phon,
            self.phon_config.step_phon
        )


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
    """Service for handling file exports - now using domain models."""

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

    def export_target_response(self, transition: PhonTransition,
                               curves: EqualLoudnessCurves,
                               filename: str) -> None:
        """Export target response grid to CSV using domain models."""
        if not self.config.export_csv:
            return

        # Use domain model for calculations - business logic removed from this layer
        relative_gains_db = curves.compute_relative_gains(
            transition.source_phon, transition.target_phon,
            use_smoothing=self.config.use_smoothing,
            smooth_window=self.config.smooth_window
        )

        from .design import prepare_target_response
        freqs, gains = prepare_target_response(
            relative_gains_db, curves.get_frequencies(), self.config
        )

        file_path = self.config.output_dir / filename
        save_response_csv(freqs, gains, str(file_path))
        self.logger.debug("Exported target response CSV")

    def export_fir_response(self, fir: np.ndarray, sample_rate: int, filename: str) -> None:
        """Export actual FIR magnitude response to CSV."""
        if not self.config.export_fir_response:
            return

        w, h = signal.freqz(fir, worN=4096, fs=sample_rate)
        mag = np.abs(h)

        w_array = np.asarray(w)
        mag_array = np.asarray(mag)

        file_path = self.config.output_dir / filename
        save_response_csv(w_array, mag_array, str(file_path))
        self.logger.debug("Exported FIR magnitude response CSV")


class FilterService:
    """High-level service for filter generation workflow using domain models."""

    def __init__(
        self,
        config: FilterConfig,
        phon_config: PhonRangeConfig,
        logger: Optional[logging.Logger] = None,
        *,
        generation_service: Optional[FilterGenerationService] = None,
        naming_service: Optional[FileNamingService] = None,
        export_service: Optional[ExportService] = None
    ):
        self.config = config
        self.phon_config = phon_config
        self.logger = logger or logging.getLogger('fir_loudness')

        self.generation_service = (
            generation_service or FilterGenerationService(
                config,
                phon_config,
                self.logger
            )
        )
        self.naming_service = naming_service or FileNamingService(config)
        self.export_service = export_service or ExportService(config, self.logger)

    def execute(self) -> int:
        """Execute the complete filter generation workflow."""
        self.logger.info("Starting filter generation workflow")

        total_filters = self.generation_service.estimate_total_filters()
        self.logger.info(f"Estimated total filters: {total_filters}")

        count = 0
        for source_phon in self.generation_service.generate_phon_levels():
            self._process_single_filter(source_phon, self.phon_config.end_phon)
            count += 1
            self.logger.debug(f"Completed filter {count}/{total_filters}")

        self.logger.info(f"Filter generation completed. Generated {count} "
                         "filters")
        return count

    def _process_single_filter(self, source_phon: float,
                               target_phon: float) -> None:
        """Process a single filter generation."""
        transition = PhonTransition(source_phon, target_phon)
        self.logger.info(f"Generating filter: {transition.get_transition_description()}")

        # Generate filter
        fir = self.generation_service.generate_single_filter(
            transition.source_phon,
            transition.target_phon,
        )
        
        # Generate file names using domain model
        curve_type = self.config.curve_type
        spec = FIRDesignSpec(
            transition.source_phon,
            transition.target_phon,
            self.config.fs,
            self.config.numtaps,
            self.generation_service.equal_loudness_curves.get_frequencies(),
            [0.0],  # placeholder, actual gains calculated later
            self.config.channels,
            self.config.sample_format
        )
        
        filter_filename = spec.to_filter_filename(curve_type)
        response_filename = spec.to_response_filename(curve_type)
        fir_response_filename = spec.to_fir_response_filename(curve_type)

        # Save and export
        self.export_service.save_filter(fir, filter_filename)

        # Export responses using domain models
        curves = self.generation_service.equal_loudness_curves
        self.export_service.export_target_response(
            transition, curves, response_filename
        )
        self.export_service.export_fir_response(
            fir, self.config.fs, fir_response_filename
        )
