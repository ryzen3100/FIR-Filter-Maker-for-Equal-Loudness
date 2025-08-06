"""
Service for phon level operations and configuration utilities.
Handles phon range operations that were moved from configuration objects.
"""

from typing import List
from .interpolation import round_phon_key


class PhonService:
    """
    Service for handling phon level operations and range calculations.
    Moved from config layer to service layer for proper separation.
    """
    
    @staticmethod
    def generate_phon_levels(start_phon: float, end_phon: float, step_phon: float):
        """
        Generate phon levels based on range configuration.
        
        Args:
            start_phon: Starting phon level
            end_phon: Ending phon level
            step_phon: Step size
            
        Yields:
            Individual phon levels in the specified range
        """
        start = round_phon_key(start_phon)
        end = round_phon_key(end_phon)
        step = round_phon_key(step_phon)
        
        if start > end:
            start, end = end, start
            
        n_steps = int(round((end - start) / step))
        for i in range(n_steps + 1):
            yield start + i * step
    
    @staticmethod
    def estimate_total_filters(start_phon: float, end_phon: float, step_phon: float) -> int:
        """Estimate total number of filters to be generated."""
        start = round_phon_key(start_phon)
        end = round_phon_key(end_phon)
        step = round_phon_key(step_phon)
        
        if start > end:
            start, end = end, start
            
        if step <= 0:
            return 1  # Handle edge case
            
        n_steps = int(round((end - start) / step))
        return n_steps + 1
    
    @staticmethod
    def get_interpolated_phons_for_range(start_phon: float, end_phon: float, 
                                       step: float) -> List[float]:
        """
        Get complete list of interpolated phon levels for a range.
        
        Args:
            start_phon: Starting phon level
            end_phon: Ending phon level  
            step: Step increment
            
        Returns:
            List of phon levels from start to end
        """
        start = round_phon_key(start_phon)
        end = round_phon_key(end_phon)
        step = round_phon_key(step)
        
        if start > end:
            start, end = end, start
            
        if step <= 0:
            return [start, end] if start != end else [start]
            
        n_steps = int(round((end - start) / step))
        return [start + i * step for i in range(n_steps + 1)]