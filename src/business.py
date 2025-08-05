"""Business logic for FIR filter generation."""

import sys
import logging
from pathlib import Path

from .config import FilterConfig, PhonRangeConfig
from .services import FilterService


class FilterGenerator:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, config: FilterConfig, phon_config: PhonRangeConfig, logger=None):
        self.config = config
        self.phon_config = phon_config
        self.logger = logger or logging.getLogger('fir_loudness')
        self._service = FilterService(config, phon_config, logger)
    
    def execute(self) -> int:
        """Execute the complete filter generation batch."""
        return self._service.execute()