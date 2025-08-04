from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Iterable

import math
import numpy as np


def round_phon_key(x: float) -> float:
    """Robust rounding to one decimal as dictionary key (avoids float drift)."""
    return float(Decimal(x).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def create_primary_interpolated_curves(curves: Dict[int, List[float]]) -> Dict[int, List[float]]:
    """
    Create midpoint curves between existing 10-phon steps to ensure a complete 10-phon grid.
    Uses a simple 0.5 linear interpolation between lower and upper 10-phon neighbors.
    """
    primary_curves: Dict[int, List[float]] = {}
    int_keys = sorted(int(k) for k in curves.keys())
    if not int_keys:
        return primary_curves
    min_key, max_key = min(int_keys), max(int_keys)
    for phon in range(min_key + 10, max_key, 20):
        lower_phon = phon - 10
        upper_phon = phon + 10
        if lower_phon in curves and upper_phon in curves:
            weight = 0.5  # midpoint
            lower_curve = np.array(curves[lower_phon], dtype=float)
            upper_curve = np.array(curves[upper_phon], dtype=float)
            interpolated_curve = lower_curve * (1 - weight) + upper_curve * weight
            primary_curves[phon] = interpolated_curve.tolist()
    return primary_curves


def create_fine_interpolated_curves(curves: Dict[int, List[float]],
                                    iso_freq: List[float],
                                    step: float = 0.1,
                                    needed_range: Optional[tuple[float, float]] = None) -> Dict[float, List[float]]:
    """
    Build fine phon curves by linear interpolation between 10-phon base curves.
    Optionally restrict to a [start, end] phon range to save time/memory.
    """
    fine_curves: Dict[float, List[float]] = {}
    # Merge curves with primaries to ensure all 10-phon steps exist
    primary = create_primary_interpolated_curves(curves)
    base = {**curves, **primary}
    base_int_keys: Dict[int, List[float]] = {int(k): v for k, v in base.items()}

    # Determine iteration range
    start = 0.0
    end = 100.0
    if needed_range is not None:
        start = max(0.0, min(100.0, needed_range[0]))
        end = max(0.0, min(100.0, needed_range[1]))
        if start > end:
            start, end = end, start

    # Integer step count to avoid drift
    n_steps = int(round((end - start) / step))
    for i in range(n_steps + 1):
        phon = start + i * step
        rounded_phon = round_phon_key(phon)
        base_key = int(rounded_phon)
        if base_key in base_int_keys and abs(rounded_phon - base_key) < 1e-9:
            fine_curves[rounded_phon] = [float(x) for x in base_int_keys[base_key]]
        else:
            interpolated_curve: List[float] = []
            lower_phon = int(math.floor(rounded_phon / 10.0) * 10)
            upper_phon = min(lower_phon + 10, 100)
            weight = (rounded_phon - lower_phon) / 10.0
            lower_curve = base_int_keys[lower_phon]
            upper_curve = base_int_keys[upper_phon]
            for idx, _freq in enumerate(iso_freq):
                # direct indexing (iso_freq and curves share indices)
                lower_val = lower_curve[idx]
                upper_val = upper_curve[idx]
                interpolated_value = lower_val * (1 - weight) + upper_val * weight
                interpolated_curve.append(float(np.round(interpolated_value, 4)))
            fine_curves[rounded_phon] = interpolated_curve
    return fine_curves
