"""
Clean application layer service using the new domain architecture.
This replaces the problematic parts of the legacy services while maintaining API compatibility.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import logging

from .config import FilterConfig, PhonRangeConfig
from .repositories import CurveRepositoryFactory
from .design import design_fir_filter_from_phon_levels, prepare_target_response
from .io_utils import save_filter_to_wav, save_response_csv
from .phon_service import PhonService
from .domain.equal_loudness_curves import EqualLoudnessCurves
from .domain.phon_transition import PhonTransition
from scipy import signal


class ApplicationService:
    """
    Clean application service using proper domain architecture.
    Handles the complete filter generation workflow with proper separation of concerns.
    """

    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig, 
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.phon_config = phon_config
        self.logger = logger or logging.getLogger('fir_loudness')
        self.phon_service = PhonService()
        
    def execute(self) -> int:
        """Execute the complete filter generation workflow."""
        self.logger.info("Starting filter generation workflow")
        
        # Initialize curves
        repo = CurveRepositoryFactory.create_repository(
            self.config.curve_type,
            self.config.iso_version
        )
        
        curves = repo.get_curves()
        frequencies = repo.get_frequencies()
        
        # Convert to domain model
        curves_dict = {float(k): v.tolist() for k, v in curves.items()}
        frequencies_list = frequencies.tolist()
        
        domain_curves = EqualLoudnessCurves(curves_dict, frequencies_list)
        domain_curves.initialize_fine_curves(
            self.phon_config.start_phon,
            self.phon_config.end_phon,
            step=0.1
        )
        
        # Generate phon levels using service
        phon_levels = list(self.phon_service.generate_phon_levels(
            self.phon_config.start_phon,
            self.phon_config.end_phon,
            self.phon_config.step_phon
        ))
        
        total_filters = len(phon_levels)
        self.logger.info(f"Generated {total_filters} phon levels")
        
        count = 0
        for source_phon in phon_levels:
            target_phon = self.phon_config.end_phon
            
            count += 1
            self.logger.info(f"Processing {count}/{total_filters}: {source_phon:.1f} → {target_phon:.1f} phon")
            
            try:
                self._process_single_filter(source_phon, target_phon, domain_curves)
                self.logger.debug(f"Completed filter {count}/{total_filters}")
            except Exception as e:
                self.logger.error(f"Failed processing {source_phon:.1f} → {target_phon:.1f}: {e}")
                raise

        self.logger.info(f"Filter generation completed. Generated {count} filters")
        return count

    def _process_single_filter(self, source_phon: float, target_phon: float, 
                             curves: EqualLoudnessCurves) -> None:
        """Process a single filter generation."""
        transition = PhonTransition(source_phon, target_phon)
        
        # Compute relative gains
        relative_gains_db = curves.compute_relative_gains(
            source_phon, target_phon,
            use_smoothing=self.config.use_smoothing,
            smooth_window=self.config.smooth_window
        )
        
        # Prepare target response
        freqs, gains = prepare_target_response(
            relative_gains_db, curves.get_frequencies(), self.config
        )
        
        # Design FIR filter
        fine_curves = curves._fine_curves
        curves_dict = fine_curves or {}
        fir = design_fir_filter_from_phon_levels(
            source_phon, target_phon,
            curves_dict,
            curves.get_frequencies(),
            self.config
        )
        
        # Generate filenames
        curve_type = self.config.curve_type
        base_filename = self._generate_filename(source_phon, target_phon, curve_type)
        
        # Save filter
        filter_filename = f"{base_filename}_filter.wav"
        save_filter_to_wav(
            fir, str(self.config.output_dir / filter_filename), 
            self.config.fs, self.config.channels, self.config.sample_format
        )
        
        # Export target response if requested
        if self.config.export_csv:
            response_filename = f"{base_filename}_response.csv"
            save_response_csv(freqs, gains, str(self.config.output_dir / response_filename))
        
        # Export FIR response if requested
        if self.config.export_fir_response:
            w, h = signal.freqz(fir, worN=4096, fs=self.config.fs)
            mag = np.abs(h)
            w_array = np.asarray(w)
            mag_array = np.asarray(mag)
            
            fir_response_filename = f"{base_filename}_fir_mag.csv"
            save_response_csv(w_array, mag_array, str(self.config.output_dir / fir_response_filename))
    
    def _generate_filename(self, source_phon: float, target_phon: float, curve_type: str) -> str:
        """Generate consistent filename base."""
        curve_map = {
            "fletcher": "FLETCHER",
            "iso2003": "ISO2003", 
            "iso2023": "ISO2023"
        }
        curve_identifier = curve_map.get(curve_type, "UNKNOWN")
        
        return f"{curve_identifier}_fs{self.config.fs}_t{self.config.numtaps}_{source_phon:.1f}-{target_phon:.1f}"