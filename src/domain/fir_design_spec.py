"""
Domain model for FIR filter design specifications.
Pure domain representation of filter requirements.
"""

from typing import Optional, Dict
import numpy as np


class FIRDesignSpec:
    """
    Value object representing complete FIR filter design requirements.
    Completely isolated from infrastructure concerns.
    """
    
    def __init__(self, 
                 source_phon: float,
                 target_phon: float,
                 sample_rate: int,
                 num_taps: int,
                 frequencies: list,
                 gains: list,
                 channels: int = 1,
                 sample_format: str = "float32"):
        """
        Initialize design specification.
        
        Args:
            source_phon: Source phon level
            target_phon: Target phon level  
            sample_rate: Sampling rate in Hz
            num_taps: Number of FIR taps
            frequencies: Frequency points list
            gains: Corresponding gain values (linear scale)
            channels: Number of audio channels
            sample_format: Output format ('float32' or 'pcm16')
        """
        self.source_phon = float(source_phon)
        self.target_phon = float(target_phon)
        self.sample_rate = int(sample_rate)
        self.num_taps = int(num_taps)
        self.frequencies = frequencies
        self.gains = gains
        self.channels = int(channels)
        self.sample_format = sample_format
        
        self._validate()
    
    def _validate(self) -> None:
        """Basic domain validation for design specification."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.num_taps <= 0:
            raise ValueError("Number of taps must be positive")
        if len(self.frequencies) != len(self.gains):
            raise ValueError("Frequencies and gains must have same length")
        if self.channels not in [1, 2]:
            raise ValueError("Channels must be 1 or 2")
        if self.sample_format not in ["float32", "pcm16"]:
            raise ValueError("Sample format must be 'float32' or 'pcm16'")
    
    def to_filter_filename(self, curve_type: str) -> str:
        """Generate consistent filename based on design spec."""
        curve_map = {
            "fletcher": "FLETCHER",
            "iso2003": "ISO2003", 
            "iso2023": "ISO2023"
        }
        curve_identifier = curve_map.get(curve_type, "UNKNOWN")
        
        return (f"{curve_identifier}_fs{self.sample_rate}_t{self.num_taps}_"
                f"{self.source_phon:.1f}-{self.target_phon:.1f}_filter.wav")
    
    def to_response_filename(self, curve_type: str) -> str:
        """Generate response CSV filename."""
        base = self.to_filter_filename(curve_type)
        return base.replace("_filter.wav", "_response.csv")
    
    def to_fir_response_filename(self, curve_type: str) -> str:
        """Generate FIR magnitude response filename."""
        base = self.to_filter_filename(curve_type)
        return base.replace("_filter.wav", "_fir_mag.csv")
    
    def get_nyquist_frequency(self) -> float:
        """Get Nyquist frequency (fs/2)."""
        return self.sample_rate / 2.0
    
    def get_frequency_range(self) -> tuple:
        """Get effective frequency range for filter design."""
        if not self.frequencies:
            return (0.0, self.get_nyquist_frequency())
        return (min(self.frequencies), max(self.frequencies))
    
    def create_design_inputs(self) -> dict:
        """
        Create dictionary suitable for FIR filter design.
        Returns dict compatible with scipy.signal.firwin2.
        """
        return {
            'freqs': np.array([0.0] + self.frequencies + [self.get_nyquist_frequency()]),
            'gains': np.array([self.gains[0]] + self.gains + [self.gains[-1]]),
            'fs': self.sample_rate
        }
    
    def get_transition_description(self) -> str:
        """Human-readable description of this transition."""
        return f"{self.source_phon:.1f} → {self.target_phon:.1f} phon ({self.sample_rate}Hz, {self.num_taps} taps)"
    
    def __str__(self) -> str:
        return self.get_transition_description()
    
    def __repr__(self) -> str:
        return (f"FIRDesignSpec({self.source_phon:.1f}→{self.target_phon:.1f} phon, "
                f"{self.sample_rate}Hz, {self.num_taps} taps)")