"""
Domain model for equal-loudness curves with interpolation and validation.
Pure domain logic without infrastructure dependencies.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from ..interpolation import create_fine_interpolated_curves, round_phon_key


class EqualLoudnessCurves:
    """
    Domain model for equal-loudness curves with interpolation capabilities.
    Handles pure business logic for curve calculations and transformations.
    """
    
    def __init__(self, curves: Dict[float, List[float]], frequencies: List[float]):
        """
        Initialize with raw curve data from repository.
        
        Args:
            curves: Dictionary mapping phon levels to dB values at frequencies
            frequencies: List of frequency points in Hz
        """
        self._curves = curves
        self._frequencies = frequencies
        self._fine_curves: Optional[Dict[float, List[float]]] = None
        self._initialized_range: Optional[Tuple[float, float]] = None
    
    def initialize_fine_curves(self, start_phon: float, end_phon: float, step: float = 0.1) -> None:
        """
        Prepare fine-grained interpolation for the specified phon range.
        
        Args:
            start_phon: Starting phon level
            end_phon: Ending phon level  
            step: Step size for interpolation
        """
        if (self._initialized_range is not None and 
            self._initialized_range[0] == start_phon and 
            self._initialized_range[1] == end_phon):
            return  # Already initialized for this range
            
        # Convert to lists for compatibility with interpolation module
        curves_as_lists = {
            k: [float(x) for x in v]
            for k, v in self._curves.items()
        }
        freqs_list = [float(x) for x in self._frequencies]
        
        self._fine_curves = create_fine_interpolated_curves(
            curves_as_lists,
            freqs_list,
            step=step,
            needed_range=(start_phon, end_phon),
        )
        self._initialized_range = (start_phon, end_phon)
    
    def get_curve_at_phon(self, phon: float) -> List[float]:
        """
        Get interpolated curve for a specific phon level.
        Must call initialize_fine_curves first for the target range.
        
        Args:
            phon: Phon level to get
            
        Returns:
            List of dB values at corresponding frequencies
        """
        phon_key = round_phon_key(phon)
        
        if self._fine_curves is None:
            raise ValueError("Fine curves not initialized. Call initialize_fine_curves first.")
            
        # Find closest available phon level
        available_phons = sorted(self._fine_curves.keys())
        if not available_phons:
            raise ValueError("No fine curves available")
            
        closest_phon = min(available_phons, key=lambda x: abs(x - phon_key))
        
        return self._fine_curves[closest_phon]
    
    def get_frequencies(self) -> List[float]:
        """Get the frequency points for these curves."""
        return self._frequencies.copy()
    
    def compute_relative_gains(self, source_phon: float, target_phon: float, 
                              use_smoothing: bool = False, 
                              smooth_window: int = 3) -> np.ndarray:
        """
        Compute relative gains needed to transform between phon levels.
        
        Args:
            source_phon: Source phon level
            target_phon: Target phon level
            use_smoothing: Whether to apply smoothing to dB values
            smooth_window: Window size for smoothing
            
        Returns:
            Array of relative gains in dB, normalized to 0 dB at 1 kHz
        """
        source_curve = np.array(self.get_curve_at_phon(source_phon))
        target_curve = np.array(self.get_curve_at_phon(target_phon))
        
        # Calculate difference
        db_diff = target_curve - source_curve
        
        # Reference to 1 kHz
        if 1000 not in self._frequencies:
            raise ValueError("Frequencies must contain 1000 Hz reference point")
            
        reference_index = self._frequencies.index(1000)
        reference_db_diff = db_diff[reference_index]
        relative_gains_db = db_diff - reference_db_diff
        
        # Apply smoothing if requested
        if use_smoothing and smooth_window > 1:
            from ..design import apply_db_smoothing
            relative_gains_db = apply_db_smoothing(relative_gains_db, smooth_window)
        
        return relative_gains_db
    
    def get_available_phons(self) -> List[float]:
        """Get sorted list of available phon levels."""
        return sorted(self._curves.keys())
    
    def get_interpolated_phons_for_range(self, start_phon: float, end_phon: float, 
                                       step: float) -> List[float]:
        """
        Generate interpolated phon levels for a range.
        
        Args:
            start_phon: Starting phon level
            end_phon: Ending phon level
            step: Step increment
            
        Returns:
            List of interpolated phon levels
        """
        start = round_phon_key(start_phon)
        end = round_phon_key(end_phon)
        
        if start > end:
            start, end = end, start
            
        n_steps = int(round((end - start) / step))
        return [start + i * step for i in range(n_steps + 1)]