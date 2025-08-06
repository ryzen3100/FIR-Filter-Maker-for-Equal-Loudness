"""Validation utilities for the FIR filter maker."""

import pathlib
from typing import Union, Any
from abc import ABC, abstractmethod


class ValidationError(ValueError):
    """Exception raised for validation errors."""
    pass


# Validation constants
SAMPLING_RATE_MIN = 8000
SAMPLING_RATE_MAX = 192000
FILTER_TAPS_MIN = 4
FILTER_TAPS_MAX = 2**18  # 262144
PHON_LEVEL_MIN = 0.0
PHON_LEVEL_MAX = 100.0
STEP_SIZE_MIN = 0.0
STEP_SIZE_MAX = 10.0
CHANNELS_MIN = 1
CHANNELS_MAX = 2
SMOOTH_WINDOW_MIN = 1
SMOOTH_WINDOW_MAX = 15
GRID_POINTS_MIN = 64
GRID_POINTS_MAX = 8192
NYQUIST_GAIN_MIN = -120.0
NYQUIST_GAIN_MAX = 120.0


class Validator(ABC):
    """Base class for parameter validation."""

    @abstractmethod
    def validate(self, value, param_name: str = "parameter") -> Any:
        """Validate a parameter value."""
        pass


class RangeValidator(Validator):
    """Validator for numeric ranges."""

    def __init__(self, min_val, max_val, value_type=None):
        self.min_val = min_val
        self.max_val = max_val
        self.value_type = value_type

    def validate(self, value, param_name: str = "parameter"):
        if self.value_type and not isinstance(value, self.value_type):
            raise ValidationError(
                f"{param_name} must be a {self.value_type.__name__}, "
                f"got {type(value)}"
            )
        if value < self.min_val or value > self.max_val:
            raise ValidationError(
                f"{param_name} {value} is out of valid range "
                f"{self.min_val}-{self.max_val}"
            )
        return value


def validate_sampling_rate(fs: int) -> int:
    """Validate sampling rate parameter."""
    validator = RangeValidator(SAMPLING_RATE_MIN, SAMPLING_RATE_MAX, int)
    return validator.validate(fs, "Sampling rate")


def validate_filter_taps(numtaps: int) -> int:
    """Validate FIR filter taps parameter."""
    validator = RangeValidator(FILTER_TAPS_MIN, FILTER_TAPS_MAX, int)
    return validator.validate(numtaps, "Filter taps")


class ChoiceValidator(Validator):
    """Validator for choices/enum values."""

    def __init__(self, valid_choices, case_sensitive=False):
        self.valid_choices = set(valid_choices)
        self.case_sensitive = case_sensitive

    def validate(self, value, param_name: str = "parameter"):
        value_str = str(value)

        if not self.case_sensitive:
            value_str = value_str.lower()
            for choice in self.valid_choices:
                if str(choice).lower() == value_str:
                    return choice
        else:
            for choice in self.valid_choices:
                if str(choice) == value_str:
                    return choice

        raise ValidationError(
            f"{param_name} must be one of {list(self.valid_choices)}, "
            f"got {value}"
        )


def validate_phon_level(phon: float, name: str = "Phon level") -> float:
    """Validate phon level parameter."""
    validator = RangeValidator(PHON_LEVEL_MIN, PHON_LEVEL_MAX, (int, float))
    return validator.validate(phon, name)


def validate_step_size(step: float) -> float:
    """Validate phon step size parameter."""
    validator = RangeValidator(STEP_SIZE_MIN, STEP_SIZE_MAX, (int, float))
    return validator.validate(step, "Step size")


def validate_channels(channels: int) -> int:
    """Validate channels parameter."""
    validator = ChoiceValidator([1, 2])
    result = validator.validate(channels, "Channels")
    return int(result)


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


def validate_directory_path(path: Union[str, pathlib.Path],
                            create: bool = True) -> pathlib.Path:
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


class OptionalRangeValidator(Validator):
    """Validator for optional numeric ranges."""

    def __init__(self, min_val, max_val, allow_none=True):
        self.min_val = min_val
        self.max_val = max_val
        self.allow_none = allow_none

    def validate(self, value, param_name: str = "parameter") -> float | None:
        if value is None and self.allow_none:
            return None

        value = float(value)
        if value < self.min_val or value > self.max_val:
            raise ValidationError(
                f"{param_name} {value} is out of valid range "
                f"[{self.min_val}, {self.max_val}]"
            )
        return value


def validate_sample_format(format_str: str) -> str:
    """Validate sample format parameter."""
    validator = ChoiceValidator(["float32", "pcm16"])
    return validator.validate(format_str, "Sample format")


def validate_dc_gain_mode(mode: str) -> str:
    """Validate DC gain mode parameter."""
    validator = ChoiceValidator(["first_iso", "unity"])
    return validator.validate(mode, "DC gain mode")


def validate_nyq_gain_db(nyq_gain: Union[float, None]) -> Union[float, None]:
    """Validate Nyquist gain parameter."""
    validator = OptionalRangeValidator(NYQUIST_GAIN_MIN, NYQUIST_GAIN_MAX)
    return validator.validate(nyq_gain, "Nyquist gain")


def validate_iso_version(iso: str) -> str:
    """Validate ISO version parameter."""
    validator = ChoiceValidator(["2003", "2023"])
    return str(validator.validate(iso, "ISO version"))


def validate_curve_type(curve: str) -> str:
    """Validate curve type parameter."""
    validator = ChoiceValidator(["iso2003", "iso2023", "fletcher"])
    return str(validator.validate(curve, "Curve type"))


def validate_grid_points(points: int) -> int:
    """Validate grid points parameter."""
    validator = RangeValidator(GRID_POINTS_MIN, GRID_POINTS_MAX, int)
    return validator.validate(points, "Grid points")
