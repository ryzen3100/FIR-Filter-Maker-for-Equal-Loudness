"""Fletcher-Munson equal-loudness contour data."""

from typing import List, Dict

FLETCHER_MUNSON_FREQ: List[float] = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 11300, 16000]

FLETCHER_MUNSON_CURVES: Dict[int, List[float]] = {
    0: [48.7, 32.5, 14.7, 2.3, 0, -1.6, 1.3, 2.5, -0.1, 4],
    10: [57.7, 42.3, 24.3, 14, 10, 7.6, 9.1, 10.2, 7.7, 13.3],
    20: [64.7, 50.6, 33.6, 25.1, 20, 17, 17.3, 18.3, 16, 22.7],
    40: [74.1, 63.9, 51.5, 46.3, 40, 36.4, 35.2, 35.9, 34.1, 41.8],
    60: [80.7, 74.8, 68.4, 65.7, 60, 56.9, 55, 55.4, 54.1, 61.4],
    80: [88, 85.9, 84.3, 83.5, 80, 78.3, 76.7, 76.8, 76, 81.4],
    100: [99.6, 99.7, 99.2, 99.5, 100, 100.6, 100.4, 100, 100, 101.8]
}

def get_fletcher_munson_data():
    """Return Fletcher-Munson frequency and curve data."""
    return FLETCHER_MUNSON_FREQ, FLETCHER_MUNSON_CURVES