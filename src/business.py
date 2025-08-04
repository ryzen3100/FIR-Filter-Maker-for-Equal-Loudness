"""Business logic for FIR filter generation."""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

from .config import FilterConfig, PhonRangeConfig
from .design import design_fir_filter_from_phon_levels
from .iso_data import ISO_FREQ, select_iso
from .interpolation import create_fine_interpolated_curves, round_phon_key
from .io_utils import save_filter_to_wav, save_response_csv, _progress_iter


class FilterGenerator:
    """Orchestrates FIR filter generation with memory efficient processing."""
    
    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig):
        self.config = config
        self.phon_config = phon_config
        self.curves = None
        self.fine_curves = None
        
    def initialize_curves(self) -> None:
        """Load and prepare ISO curves once."""
        self.curves = select_iso(self.config.iso_version)
        
        # Create fine interpolation curves on-demand with memory optimization
        s = round_phon_key(self.phon_config.start_phon)
        e = round_phon_key(self.phon_config.end_phon)
        self.fine_curves = create_fine_interpolated_curves(
            self.curves, ISO_FREQ, step=0.1, needed_range=(s, e)
        )
        
    def generate_single_filter(self, source_phon: float, target_phon: float) -> np.ndarray:
        """Generate a single FIR filter."""
        if self.fine_curves is None:
            self.initialize_curves()
            
        return design_fir_filter_from_phon_levels(
            source_phon, target_phon, self.fine_curves, ISO_FREQ, self.config
        )
        
    def generate_file_name(self, source_phon: float, target_phon: float) -> str:
        """Generate consistent file name for filter."""
        iso_tag = f"ISO{self.config.iso_version}"
        return f"{iso_tag}_fs{self.config.fs}_t{self.config.numtaps}_{source_phon:.1f}-{target_phon:.1f}_filter.wav"
        
    def process_single_filter(self, source_phon: float, target_phon: float) -> None:
        """Process a single filter generation."""
        # Generate filter
        fir = self.generate_single_filter(source_phon, target_phon)
        
        # Generate file name
        base_name = self.generate_file_name(source_phon, target_phon)
        wav_path = self.config.output_dir / base_name
        
        # Save filter to WAV
        save_filter_to_wav(
            fir, str(wav_path), self.config.fs,
            channels=self.config.channels,
            sample_format=self.config.sample_format
        )
        
        # Export response CSV if requested
        if self.config.export_csv:
            self._export_response_csv(source_phon, target_phon, fir, wav_path)
            
        # Export FIR magnitude response if requested
        if self.config.export_fir_response:
            self._export_fir_response(source_phon, target_phon, fir)
    
    def _export_response_csv(self, source_phon: float, target_phon: float, fir: np.ndarray, wav_path: Path) -> None:
        """Export target response grid to CSV."""
        from .design import compute_relative_gains, prepare_target_response
        
        relative_gains_db = compute_relative_gains(
            self.fine_curves, source_phon, target_phon, ISO_FREQ,
            self.config.use_smoothing, self.config.smooth_window
        )
        
        freqs, gains = prepare_target_response(relative_gains_db, ISO_FREQ, self.config)
        
        csv_path = self.config.output_dir / f"{wav_path.stem.replace('_filter', '_response')}.csv"
        save_response_csv(freqs, gains, str(csv_path))
    
    def _export_fir_response(self, source_phon: float, target_phon: float, fir: np.ndarray) -> None:
        """Export actual FIR magnitude response to CSV."""
        from scipy import signal
        
        w, h = signal.freqz(fir, worN=4096, fs=self.config.fs)
        mag = np.abs(h)
        
        base_name = self.generate_file_name(source_phon, target_phon)
        csv_path = self.config.output_dir / f"{base_name.replace('_filter.wav', '_fir_mag.csv')}"
        save_response_csv(w, mag, str(csv_path))
        
    def execute(self) -> int:
        """
        Execute the complete filter generation batch.
        
        Returns:
            Number of filters generated
        """
        if self.fine_curves is None:
            self.initialize_curves()
        
        total_filters = self.phon_config.estimate_total_filters()
        progress_enabled = sys.stdout.isatty()
        
        count = 0
        for source_phon in _progress_iter(
            list(self.phon_config.generate_phon_levels()),
            total_filters,
            progress_enabled,
            desc="Generating filters"
        ):
            self.process_single_filter(source_phon, self.phon_config.end_phon)
            count += 1
        
        return count