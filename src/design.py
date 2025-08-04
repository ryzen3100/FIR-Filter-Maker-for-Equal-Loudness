from typing import Dict, List, Optional, Literal

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

from .interpolation import round_phon_key


def design_fir_filter_from_phon_levels(phon1: float,
                                       phon2: float,
                                       numtaps: int,
                                       fs: int,
                                       fine_curves: Dict[float, List[float]],
                                       iso_freq: List[float],
                                       smooth_db: bool = False,
                                       smooth_window: int = 3,
                                       dc_gain_mode: Literal["first_iso", "unity"] = "first_iso",
                                       nyq_gain_db: Optional[float] = None,
                                       grid_points: int = 2048) -> np.ndarray:
    """
    Design an FIR filter to transform equal-loudness curve at phon1 to curve at phon2.
    The response is normalized to 0 dB at 1 kHz. Optional smoothing in dB domain.
    """
    p1 = round_phon_key(phon1)
    p2 = round_phon_key(phon2)
    if p1 not in fine_curves or p2 not in fine_curves:
        raise ValueError(f"Phon keys not available: {p1}, {p2}")

    db_diff = np.array(fine_curves[p2], dtype=float) - np.array(fine_curves[p1], dtype=float)

    # Reference to 1 kHz (ensure index exists)
    if 1000 not in iso_freq:
        raise ValueError("iso_freq must contain 1000 Hz reference")
    reference_db_diff = db_diff[iso_freq.index(1000)]
    relative_gains_db = db_diff - reference_db_diff

    # Optional smoothing in dB to avoid sharp transitions
    if smooth_db and smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
        pad = np.pad(relative_gains_db, (smooth_window // 2, smooth_window - 1 - smooth_window // 2), mode='edge')
        relative_gains_db = np.convolve(pad, kernel, mode='valid')

    relative_gains_linear = 10.0 ** (relative_gains_db / 20.0)

    # Prepare interpolation nodes including DC and Nyquist with configurable edges
    nyquist = fs / 2.0
    iso_nodes_hz = np.array([0.0] + iso_freq + [nyquist], dtype=float)

    # DC gain choice
    if dc_gain_mode == "unity":
        dc_gain = 1.0
    else:
        dc_gain = float(relative_gains_linear[0])

    # Nyquist gain choice
    if nyq_gain_db is None:
        nyq_gain = 0.0  # default taper to 0 (to mitigate HF ringing)
    else:
        nyq_gain = float(10.0 ** (nyq_gain_db / 20.0))

    gain_nodes_linear = np.array([dc_gain] + list(relative_gains_linear) + [nyq_gain], dtype=float)

    # Use PCHIP (monotone) to avoid overshoot
    interpolator = PchipInterpolator(iso_nodes_hz, gain_nodes_linear, extrapolate=False)

    # Frequency grid for firwin2 (decoupled from numtaps for performance)
    grid_points = max(16, int(grid_points))
    freqs = np.linspace(0.0, nyquist, grid_points, dtype=float)
    gains = interpolator(freqs)
    # Safety: replace NaNs from extrapolation with edge values
    if np.isnan(gains).any():
        gains = np.nan_to_num(gains, nan=gain_nodes_linear[-1])

    # Ensure endpoints are exactly as intended
    gains[0] = dc_gain
    gains[-1] = nyq_gain

    # Design FIR
    fir = signal.firwin2(numtaps, freqs, gains, fs=fs)
    return fir
