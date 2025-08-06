"""
Domain model for phon level transitions.
Handles pure business logic for phon transformations.
"""

from typing import Optional, Tuple

from .equal_loudness_curves import EqualLoudnessCurves


class PhonTransition:
    """
    Value object representing a phon level transition.
    Contains business logic for calculating transformations between phon levels.
    """
    
    def __init__(self, source_phon: float, target_phon: float):
        """
        Initialize transition with source and target phon levels.
        
        Args:
            source_phon: Starting phon level
            target_phon: Ending phon level
        """
        self.source_phon = float(source_phon)
        self.target_phon = float(target_phon)
        
        if self.source_phon < 0 or self.target_phon < 0:
            raise ValueError("Phon levels must be non-negative")
        if self.source_phon > 100 or self.target_phon > 100:
            raise ValueError("Phon levels must be <= 100")
    
    def validate_with_curves(self, curves: EqualLoudnessCurves) -> None:
        """
        Validate that source and target phon levels exist in the curves.
        
        Args:
            curves: EqualLoudnessCurves instance to validate against
            
        Raises:
            ValueError: If phon levels are not available in curves
        """
        available_phons = curves.get_available_phons()
        
        # Validate rough bounds (fine curves will be interpolated)
        if self.source_phon < min(available_phons) or self.source_phon > max(available_phons):
            raise ValueError(f"Source phon {self.source_phon} outside available range")
            
        if self.target_phon < min(available_phons) or self.target_phon > max(available_phons):
            raise ValueError(f"Target phon {self.target_phon} outside available range")
    
    def calculate_relative_gains(self, curves: EqualLoudnessCurves,
                               use_smoothing: bool = False,
                               smooth_window: int = 3) -> list:
        """
        Calculate relative gains for this specific transition.
        
        Args:
            curves: EqualLoudnessCurves instance with initialized fine curves
            use_smoothing: Whether to apply smoothing to dB values
            smooth_window: Window size for smoothing
            
        Returns:
            List of relative gains in dB
        """
        return curves.compute_relative_gains(
            self.source_phon,
            self.target_phon,
            use_smoothing=use_smoothing,
            smooth_window=smooth_window
        ).tolist()
    
    def get_transition_description(self) -> str:
        """Get human-readable description of the transition."""
        return f"{self.source_phon:.1f} → {self.target_phon:.1f} phon"
    
    def get_phon_change(self) -> float:
        """Get the absolute phon level change."""
        return abs(self.target_phon - self.source_phon)
    
    def is_louder(self) -> bool:
        """Check if transition goes to louder levels."""
        return self.target_phon > self.source_phon
    
    def __str__(self) -> str:
        return self.get_transition_description()
    
    def __repr__(self) -> str:
        return f"PhonTransition({self.source_phon:.1f} -> {self.target_phon:.1f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PhonTransition):
            return False
        return (self.source_phon == other.source_phon and 
                self.target_phon == other.target_phon)
    
    def __hash__(self) -> int:
        return hash((self.source_phon, self.target_phon))