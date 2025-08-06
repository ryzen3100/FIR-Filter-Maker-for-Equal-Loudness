"""Repository pattern for data access."""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from .iso_data import ISO_FREQ, select_iso
from .fletcher_data import get_fletcher_munson_data


class CurveRepository(ABC):
    """Abstract base class for curve data repositories."""

    @abstractmethod
    def get_curves(self) -> Dict[float, np.ndarray]:
        """Get equal-loudness curves data."""
        pass

    @abstractmethod
    def get_frequencies(self) -> np.ndarray:
        """Get frequency points for the curves."""
        pass

    @abstractmethod
    def get_curve_type(self) -> str:
        """Get the type of curves provided."""
        pass


class ISORepository(CurveRepository):
    """Repository for ISO 226 equal-loudness contours."""

    def __init__(self, version: str = "2023"):
        self.version = version
        self._curves: Dict[float, np.ndarray] | None = None
        self._frequencies: np.ndarray | None = None

    def get_curves(self) -> Dict[float, np.ndarray]:
        """Get ISO 226 equal-loudness curves."""
        if self._curves is None:
            raw_curves = select_iso(self.version)
            self._curves = {float(k): np.array(v) for k, v in
                           raw_curves.items()}
        return self._curves

    def get_frequencies(self) -> np.ndarray:
        """Get ISO frequency points."""
        if self._frequencies is None:
            self._frequencies = np.array(ISO_FREQ)
        return self._frequencies

    def get_curve_type(self) -> str:
        """Get ISO curve type identifier."""
        return f"iso{self.version}"


class FletcherRepository(CurveRepository):
    """Repository for Fletcher-Munson equal-loudness contours."""

    def __init__(self):
        self._curves: Dict[float, np.ndarray] | None = None
        self._frequencies: np.ndarray | None = None

    def get_curves(self) -> Dict[float, np.ndarray]:
        """Get Fletcher-Munson equal-loudness curves."""
        if self._curves is None:
            _, curves = get_fletcher_munson_data()
            self._curves = {float(k): np.array(v) for k, v in
                           curves.items()}
        return self._curves

    def get_frequencies(self) -> np.ndarray:
        """Get Fletcher-Munson frequency points."""
        if self._frequencies is None:
            freqs, _ = get_fletcher_munson_data()
            self._frequencies = np.array(freqs)
        return self._frequencies

    def get_curve_type(self) -> str:
        """Get Fletcher curve type identifier."""
        return "fletcher"


class CurveRepositoryFactory:
    """Factory for creating curve repositories."""

    @staticmethod
    def create_repository(curve_type: str,
                          iso_version: str = "2023") -> CurveRepository:
        """Create appropriate repository based on curve type."""
        if curve_type == "fletcher":
            return FletcherRepository()
        elif curve_type.startswith("iso"):
            return ISORepository(iso_version)
        else:
            raise ValueError(f"Unsupported curve type: {curve_type}")
