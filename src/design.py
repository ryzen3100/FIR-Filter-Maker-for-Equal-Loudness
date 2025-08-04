from typing import Dict, List, Optional

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

from .config import FilterConfig
from .interpolation import round_phon_key


def apply_db_smoothing(db_values: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply smoothing to dB values using a moving average.
    
    Args:
        db_values: Array of dB values to smooth
        window_size: Size of the smoothing window (must be odd)
        
    Returns:
        Smoothed dB values
    """
    if window_size <= 1:
        return db_values
    
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    pad_width = (window_size // 2, window_size - 1 - window_size // 2)
    padded = np.pad(db_values, pad_width, mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def compute_relative_gains(
    target_curves: Dict[float, List[float]],
    source_phon: float,
    target_phon: float,
    iso_freq: List[float],
    smooth_db: bool = False,
    smooth_window: int = 3
) -> np.ndarray:
    """
    Compute relative gains needed to transform from source to target phon level.
    
    Args:
        target_curves: Dictionary mapping phon levels to dB values
        source_phon: Source phon level
        target_phon: Target phon level
        iso_freq: Frequency points for ISO curves
        smooth_db: Whether to apply smoothing to the dB values
        smooth_window: Window size for smoothing
        
    Returns:
        Array of relative gains in dB, normalized to 0 dB at 1 kHz
    """
    source_key = round_phon_key(source_phon)
    target_key = round_phon_key(target_phon)
    
    if source_key not in target_curves:
        raise ValueError(f"Source phon level {source_key} not available in curves")
    if target_key not in target_curves:
        raise ValueError(f"Target phon level {target_key} not available in curves")
    
    # Convert to numpy arrays
    source_db = np.array(target_curves[source_key], dtype=float)
    target_db = np.array(target_curves[target_key], dtype=float)
    
    # Calculate difference
    db_diff = target_db - source_db
    
    # Reference to 1 kHz
    if 1000 not in iso_freq:
        raise ValueError("iso_freq must contain 1000 Hz reference")
    
    reference_index = iso_freq.index(1000)
    reference_db_diff = db_diff[reference_index]
    relative_gains_db = db_diff - reference_db_diff
    
    # Apply smoothing if requested
    if smooth_db and smooth_window > 1:
        relative_gains_db = apply_db_smoothing(relative_gains_db, smooth_window)
    
    return relative_gains_db


def prepare_target_response(
    relative_gains_db: np.ndarray,
    iso_freq: List[float],
    config: FilterConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare frequency and gain arrays for FIR filter design.
    
    Args:
        relative_gains_db: Relative gains in dB
        iso_freq: ISO frequency points
        config: Filter configuration
        
    Returns:
        Tuple of (frequencies, gains) arrays ready for FIR design
    """
    fs = config.fs
    nyquist = fs / 2.0
    
    # Convert dB gains to linear
    relative_gains_linear = 10.0 ** (relative_gains_db / 20.0)
    
    # Prepare interpolation nodes
    iso_nodes_hz = np.array([0.0] + iso_freq + [nyquist], dtype=float)
    
    # DC gain choice
    if config.dc_gain_mode == "unity":
        dc_gain = 1.0
    else:
        dc_gain = float(relative_gains_linear[0])
    
    # Nyquist gain choice
    if config.nyq_gain_db is None:
        nyq_gain = 0.0  # default taper to 0
    else:
        nyq_gain = float(10.0 ** (config.nyq_gain_db / 20.0))
    
    gain_nodes_linear = np.array([dc_gain] + relative_gains_linear.tolist() + [nyq_gain], dtype=float)
    
    # Create interpolation grid
    grid_points = max(16, config.grid_points)
    freqs = np.linspace(0.0, nyquist, grid_points, dtype=float)
    
    # Use PCHIP interpolation
    interpolator = PchipInterpolator(iso_nodes_hz, gain_nodes_linear, extrapolate=False)
    gains = interpolator(freqs)
    
    # Handle NaN values
    if np.isnan(gains).any():
        gains = np.nan_to_num(gains, nan=nyq_gain)
    
    # Ensure endpoints are exactly as intended
    gains[0] = dc_gain
    gains[-1] = nyq_gain
    
    return freqs, gains


def design_fir_filter_from_phon_levels(
    source_phon: float,
    target_phon: float,
    target_curves: Dict[float, List[float]],
    iso_freq: List[float],
    config: FilterConfig
) -> np.ndarray:
    """
    Design an FIR filter to transform equal-loudness curve between phon levels.
    
    Args:
        source_phon: Source phon level
        target_phon: Target phon level
        target_curves: Dictionary of phon level curves
        iso_freq: ISO frequency points
        config: Filter configuration
        
    Returns:
        FIR filter coefficients as numpy array
    """
    # Compute relative gains
    relative_gains_db = compute_relative_gains(
        target_curves, source_phon, target_phon, iso_freq,
        config.use_smoothing, config.smooth_window
    )
    
    # Prepare target response
    freqs, gains = prepare_target_response(relative_gains_db, iso_freq, config)
    
    # Design FIR filter
    fir_coefficients = signal.firwin2(config.numtaps, freqs, gains, fs=config.fs)
    
    return fir_coefficients
