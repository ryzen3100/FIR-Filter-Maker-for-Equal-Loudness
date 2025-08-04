"""IO utilities with proper error handling and pathlib support."""

import csv
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.io import wavfile


class IOError(Exception):
    """Exception raised for IO errors."""
    pass


def save_filter_to_wav(
    coefficients: np.ndarray,
    filepath: str | Path,
    sampling_rate: int,
    channels: int = 1,
    sample_format: str = "float32",
    normalize_pcm16: bool = True
) -> None:
    """
    Save filter coefficients to WAV file.
    
    Args:
        coefficients: Filter coefficients
        filepath: Output file path
        sampling_rate: Audio sampling rate in Hz
        channels: 1 for mono, 2 for stereo
        sample_format: 'float32' or 'pcm16'
        normalize_pcm16: Normalize PCM16 to 0 dBFS peak
    
    Raises:
        ValueError: For invalid parameters
        IOError: For file access issues
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    
    if sample_format == "pcm16":
        arr = coefficients.astype(np.float64, copy=False)
        if normalize_pcm16:
            peak = np.max(np.abs(arr)) if arr.size > 0 else 1.0
            if peak == 0:
                peak = 1.0
            arr = arr / peak
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)
        data_to_write = _create_multichannel_data(pcm, channels)
        wavfile.write(str(filepath), sampling_rate, data_to_write)
    
    elif sample_format == "float32":
        coeff32 = coefficients.astype(np.float32, copy=False)
        data_to_write = _create_multichannel_data(coeff32, channels)
        wavfile.write(str(filepath), sampling_rate, data_to_write)
    
    else:
        raise ValueError("Unsupported sample format. Use 'float32' or 'pcm16'.")


def _create_multichannel_data(data: np.ndarray, channels: int) -> np.ndarray:
    """Create multi-channel data array for WAV export."""
    if channels == 2:
        # Create stereo array
        return np.stack([data, data], axis=-1)
    return data


def save_response_csv(
    frequencies: np.ndarray,
    gains: np.ndarray,
    filepath: str | Path
) -> None:
    """
    Save frequency response data to CSV.
    
    Args:
        frequencies: Array of frequencies in Hz
        gains: Array of gains (linear scale)
        filepath: Output file path
        
    Raises:
        IOError: For file access issues
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with filepath.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency_Hz", "Gain_linear"])
            for freq, gain in zip(frequencies, gains):
                writer.writerow([float(freq), float(gain)])
    except Exception as e:
        raise IOError(f"Failed to save CSV file {filepath}: {e}")


def _progress_iter(
    iterable: Iterable[float],
    total: Optional[int],
    enabled: bool,
    desc: str = ""
) -> Iterable[float]:
    """
    Minimal dependency-free progress indicator.
    
    Args:
        iterable: Items to iterate over
        total: Total number of items
        enabled: Whether to show progress
        desc: Description text
        
    Yields:
        Items from the iterable
    """
    if not enabled or total is None or total <= 0:
        for x in iterable:
            yield x
        return

    width = 30  # progress bar width
    count = 0
    term_cols = shutil.get_terminal_size((80, 20)).columns
    
    try:
        for x in iterable:
            count += 1
            pct = count / total
            filled = int(pct * width)
            bar = "#" * filled + "-" * (width - filled)
            msg = f"{desc} [{bar}] {count}/{total} ({pct*100:5.1f}%)"
            
            if len(msg) > term_cols - 1:
                msg = msg[: term_cols - 1]
                
            print("\r" + msg, end="", flush=True)
            yield x
            
    finally:
        # Ensure progress bar completes and moves to next line
        print("\r" + f"{desc} [{'#'*width}] {total}/{total} (100.0%)" + " " * max(0, term_cols - len(desc) - width - 20), flush=True)
        print()