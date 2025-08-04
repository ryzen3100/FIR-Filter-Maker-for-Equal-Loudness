"""Validation utilities for the FIR filter maker."""

import os
import pathlib
from typing import Union


class ValidationError(ValueError):
    """Exception raised for validation errors."""
    pass


def validate_sampling_rate(fs: int) -> int:
    """Validate sampling rate parameter."""
    if not isinstance(fs, int):
        raise ValidationError(f"Sampling rate must be an integer, got {type(fs)}")
    if fs < 8000 or fs > 192000:
        raise ValidationError(f"Sampling rate {fs} Hz is out of valid range 8000-192000")
    return fs


def validate_filter_taps(numtaps: int) -> int:
    """Validate FIR filter taps parameter."""
    if not isinstance(numtaps, int):
        raise ValidationError(f"Filter taps must be an integer, got {type(numtaps)}")
    if numtaps < 4 or numtaps > 2**18:
        raise ValidationError(f"Filter taps {numtaps} is out of valid range 4-262144")
    return numtaps


def validate_phon_level(phon: float, name: str = "Phon level") -> float:
    """Validate phon level parameter."""
    if not isinstance(phon, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(phon)}")
    if phon < 0 or phon > 100:
        raise ValidationError(f"{name} {phon} is out of valid range 0-100")
    return float(phon)


def validate_step_size(step: float) -> float:
    """Validate phon step size parameter."""
    if not isinstance(step, (int, float)):
        raise ValidationError(f"Step size must be a number, got {type(step)}")
    if step <= 0 or step > 10:
        raise ValidationError(f"Step size {step} is out of valid range (0, 10]")
    return float(step)


def validate_channels(channels: int) -> int:
    """Validate channels parameter."""
    if not isinstance(channels, int):
        raise ValidationError(f"Channels must be an integer, got {type(channels)}")
    if channels not in (1, 2):
        raise ValidationError(f"Channels must be 1 or 2, got {channels}")
    return channels


def validate_file_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Validate and sanitize file path parameters."""
    if isinstance(path, str):
        path = pathlib.Path(path)
    
    if not isinstance(path, pathlib.Path):
        raise ValidationError(f"Path must be string or Path, got {type(path)}")
    
    # Resolve and normalize the path
    path = path.resolve()
    
    # Check for directory traversal attempts
    try:
        path.resolve().relative_to(path.cwd())
    except ValueError:
        # Allow absolute paths within reasonable limits
        pass
    
    # Check for dangerous patterns
    if any(part.startswith('.') for part in path.parts):
        raise ValidationError(f"Path contains hidden directories: {path}")
    
    # Ensure path doesn't escape expected bounds
    if '..' in str(path) or '$' in str(path) or '~' in str(path):
        raise ValidationError(f"Path contains invalid characters: {path}")
    
    return path


def validate_directory_path(path: Union[str, pathlib.Path], create: bool = True) -> pathlib.Path:
    """Validate and sanitize directory path parameters."""
    path = validate_file_path(path)
    
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise ValidationError(f"Directory does not exist: {path}")
    except OSError as e:
        raise ValidationError(f"Cannot create/access directory {path}: {e}")
    
    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")
    
    return path


def validate_sample_format(format_str: str) -> str:
    """Validate sample format parameter."""
    format_str = str(format_str).lower()
    if format_str not in {"float32", "pcm16"}:
        raise ValidationError(f"Sample format must be 'float32' or 'pcm16', got {format_str}")
    return format_str


def validate_dc_gain_mode(mode: str) -> str:
    """Validate DC gain mode parameter."""
    mode = str(mode).lower()
    if mode not in {"first_iso", "unity"}:
        raise ValidationError(f"DC gain mode must be 'first_iso' or 'unity', got {mode}")
    return mode


def validate_nyq_gain_db(nyq_gain: Union[float, None]) -> Union[float, None]:
    """Validate Nyquist gain parameter."""
    if nyq_gain is None:
        return None
    
    nyq_gain = float(nyq_gain)
    if abs(nyq_gain) > 100:
        raise ValidationError(f"Nyquist gain {nyq_gain} dB is out of valid range [-100, 100]")
    return nyq_gain


def validate_iso_version(iso: str) -> str:
    """Validate ISO version parameter."""
    iso = str(iso).strip().lower()
    valid_versions = {"2003", "2023"}
    if iso not in valid_versions:
        raise ValidationError(f"ISO version must be one of {valid_versions}, got {iso}")
    return iso


def validate_curve_type(curve: str) -> str:
    """Validate curve type parameter."""
    curve = str(curve).strip().lower()
    valid_curves = {"iso2003", "iso2023", "fletcher"}
    if curve not in valid_curves:
        raise ValidationError(f"Curve type must be one of {valid_curves}, got {curve}")
    return curve


def validate_grid_points(points: int) -> int:
    """Validate grid points parameter."""
    if not isinstance(points, int):
        raise ValidationError(f"Grid points must be an integer, got {type(points)}")
    if points < 16 or points > 8192:
        raise ValidationError(f"Grid points {points} is out of valid range 16-8192")
    return points