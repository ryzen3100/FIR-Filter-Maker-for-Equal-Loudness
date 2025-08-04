#!/usr/bin/env python3
import argparse
import os
import sys
import math
import csv
import wave
import struct
import shutil
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator


# ===============================
# ISO DATASETS (2003 vs 2023)
# ===============================

ISO_FREQ = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]

ISO_CURVES_2003 = {
    0: [76.55, 65.62, 55.12, 45.53, 37.63, 30.86, 25.02, 20.51, 16.65, 13.12, 10.09, 7.54, 5.11, 3.06, 1.48, 0.3, -0.3, -0.01, 1.03, -1.19, -4.11, -7.05, -9.03, -8.49, -4.48, 3.28, 9.83, 10.48, 8.38, 14.1, 79.65],
    10: [83.75, 75.76, 68.21, 61.14, 54.96, 49.01, 43.24, 38.13, 33.48, 28.77, 24.84, 21.33, 18.05, 15.14, 12.98, 11.18, 9.99, 10, 11.26, 10.43, 7.27, 4.45, 3.04, 3.8, 7.46, 14.35, 20.98, 23.43, 22.33, 25.17, 81.47],
    20: [89.58, 82.65, 75.98, 69.62, 64.02, 58.55, 53.19, 48.38, 43.94, 39.37, 35.51, 31.99, 28.69, 25.67, 23.43, 21.48, 20.1, 20.01, 21.46, 21.4, 18.15, 15.38, 14.26, 15.14, 18.63, 25.02, 31.52, 34.43, 33.04, 34.67, 84.18],
    40: [99.85, 93.94, 88.17, 82.63, 77.78, 73.08, 68.48, 64.37, 60.59, 56.7, 53.41, 50.4, 47.58, 44.98, 43.05, 41.34, 40.06, 40.01, 41.82, 42.51, 39.23, 36.51, 35.61, 36.65, 40.01, 45.83, 51.8, 54.28, 51.49, 51.96, 92.77],
    60: [109.51, 104.23, 99.08, 94.18, 89.96, 85.94, 82.05, 78.65, 75.56, 72.47, 69.86, 67.53, 65.39, 63.45, 62.05, 60.81, 59.89, 60.01, 62.15, 63.19, 59.96, 57.26, 56.42, 57.57, 60.89, 66.36, 71.66, 73.16, 68.63, 68.43, 104.92],
    80: [118.99, 114.23, 109.65, 105.34, 101.72, 98.36, 95.17, 92.48, 90.09, 87.82, 85.92, 84.31, 82.89, 81.68, 80.86, 80.17, 79.67, 80.01, 82.48, 83.74, 80.59, 77.88, 77.07, 78.31, 81.62, 86.81, 91.41, 91.74, 85.41, 84.67, 118.95],
    100: [128.41, 124.15, 120.11, 116.38, 113.35, 110.65, 108.16, 106.17, 104.48, 103.03, 101.85, 100.97, 100.3, 99.83, 99.62, 99.5, 99.44, 100.01, 102.81, 104.25, 101.18, 98.48, 97.67, 99, 102.3, 107.23, 111.11, 110.23, 102.07, 100.83, 133.73]
}

ISO_CURVES_2023 = {
    0: [76.5517, 65.6189, 55.1228, 45.5340, 37.6321, 30.8650, 25.0238, 20.5100, 16.6458, 13.1160, 10.0883, 7.5436, 5.1137, 3.0589, 1.4824, 0.3029, -0.3026, -0.0103, 1.0335, -1.1863, -4.1116, -7.0462, -9.0260, -8.4944, -4.4829, 3.2817, 9.8291, 10.4757, 8.3813, 14.1000, 79.6500],
    10: [83.7500, 75.7579, 68.2089, 61.1365, 54.9638, 49.0098, 43.2377, 38.1338, 33.4772, 28.7734, 24.8417, 21.3272, 18.0522, 15.1379, 12.9768, 11.1791, 9.9918, 9.9996, 11.2621, 10.4291, 7.2744, 4.4508, 3.0404, 3.7961, 7.4583, 14.3483, 20.9841, 23.4306, 22.3269, 25.1700, 81.4700],
    20: [89.5781, 82.6513, 75.9764, 69.6171, 64.0178, 58.5520, 53.1898, 48.3809, 43.9414, 39.3702, 35.5126, 31.9922, 28.6866, 25.6703, 23.4263, 21.4825, 20.1011, 20.0052, 21.4618, 21.4013, 18.1515, 15.3844, 14.2559, 15.1415, 18.6349, 25.0196, 31.5227, 34.4256, 33.0444, 34.6700, 84.1800],
    40: [99.8539, 93.9444, 88.1659, 82.6287, 77.7849, 73.0825, 68.4779, 64.3711, 60.5855, 56.7022, 53.4087, 50.3992, 47.5775, 44.9766, 43.0507, 41.3392, 40.0618, 40.0100, 41.8195, 42.5076, 39.2296, 36.5090, 35.6089, 36.6492, 40.0077, 45.8283, 51.7968, 54.2841, 51.4859, 51.9600, 92.7700],
    60: [109.5113, 104.2279, 99.0779, 94.1773, 89.9635, 85.9434, 82.0534, 78.6546, 75.5635, 72.4743, 69.8643, 67.5348, 65.3917, 63.4510, 62.0512, 60.8150, 59.8867, 60.0116, 62.1549, 63.1894, 59.9616, 57.2552, 56.4239, 57.5699, 60.8882, 66.3613, 71.6640, 73.1551, 68.6308, 68.4300, 104.9200],
    80: [118.9900, 114.2326, 109.6457, 105.3367, 101.7214, 98.3618, 95.1729, 92.4797, 90.0892, 87.8162, 85.9166, 84.3080, 82.8934, 81.6786, 80.8634, 80.1736, 79.6691, 80.0121, 82.4834, 83.7408, 80.5867, 77.8847, 77.0748, 78.3124, 81.6182, 86.8087, 91.4062, 91.7361, 85.4068, 84.6700, 118.9500],
    100: [128.4100, 124.1500, 120.1100, 116.3800, 113.3500, 110.6500, 108.1600, 106.1700, 104.4800, 103.0300, 101.8500, 100.9700, 100.3000, 99.8300, 99.6200, 99.5000, 99.4400, 100.0100, 102.8100, 104.2500, 101.1800, 98.4800, 97.6700, 99.0000, 102.3000, 107.2300, 111.1100, 110.2300, 102.0700, 100.8300, 133.7300]
}


def select_iso(iso: str) -> Dict[int, List[float]]:
    iso = iso.strip().lower()
    if iso in ("2003", "iso226-2003", "iso2003"):
        return ISO_CURVES_2003
    if iso in ("2023", "iso226-2023", "iso2023"):
        return ISO_CURVES_2023
    raise ValueError("Unsupported ISO set. Use --iso {2003,2023}")


# ===============================
# CURVE INTERPOLATION
# ===============================

def create_primary_interpolated_curves(curves: Dict[int, List[float]]) -> Dict[int, List[float]]:
    primary_curves = {}
    for phon in [30, 50, 70, 90]:
        lower_phon = phon - 10
        upper_phon = phon + 10
        weight = 0.5
        lower_curve = np.array(curves[lower_phon])
        upper_curve = np.array(curves[upper_phon])
        interpolated_curve = lower_curve * (1 - weight) + upper_curve * weight
        primary_curves[phon] = interpolated_curve.tolist()
    return primary_curves


def round_phon_key(x: float) -> float:
    # Robust rounding to one decimal as dictionary key (avoids float drift)
    return float(Decimal(x).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def create_fine_interpolated_curves(curves: Dict[int, List[float]],
                                    iso_freq: List[float],
                                    step: float = 0.1) -> Dict[float, List[float]]:
    fine_curves: Dict[float, List[float]] = {}
    # Merge curves with primaries to ensure all 10-phon steps exist
    primary = create_primary_interpolated_curves(curves)
    base = {**curves, **primary}
    # Keys in base are at 10-phon integer steps only (e.g., 0,10,20,...,100)
    # Make this explicit to type-checkers and avoid float/int key confusion.
    base_int_keys: Dict[int, List[float]] = {int(k): v for k, v in base.items()}

    num_steps = int(round(100.0 / step))
    for i in range(num_steps + 1):
        phon = i * step
        rounded_phon = round_phon_key(phon)
        if int(rounded_phon) in base_int_keys:
            fine_curves[rounded_phon] = base_int_keys[int(rounded_phon)]
        else:
            interpolated_curve = []
            lower_phon = int(math.floor(rounded_phon / 10.0) * 10)
            upper_phon = min(lower_phon + 10, 100)
            weight = (rounded_phon - lower_phon) / 10.0
            for idx, freq in enumerate(iso_freq):
                lower_val = np.interp(freq, iso_freq, base_int_keys[lower_phon])
                upper_val = np.interp(freq, iso_freq, base_int_keys[upper_phon])
                interpolated_value = lower_val * (1 - weight) + upper_val * weight
                interpolated_curve.append(float(np.round(interpolated_value, 4)))
            fine_curves[rounded_phon] = interpolated_curve
    return fine_curves


# ===============================
# FILTER DESIGN
# ===============================

def design_fir_filter_from_phon_levels(phon1: float,
                                       phon2: float,
                                       numtaps: int,
                                       fs: int,
                                       fine_curves: Dict[float, List[float]],
                                       iso_freq: List[float]) -> np.ndarray:
    p1 = round_phon_key(phon1)
    p2 = round_phon_key(phon2)
    if p1 not in fine_curves or p2 not in fine_curves:
        raise ValueError(f"Phon keys not available: {p1}, {p2}")

    db_diff = np.array(fine_curves[p2]) - np.array(fine_curves[p1])

    # Reference to 1 kHz (ensure index exists)
    if 1000 not in iso_freq:
        raise ValueError("iso_freq must contain 1000 Hz reference")
    reference_db_diff = db_diff[iso_freq.index(1000)]
    relative_gains_db = db_diff - reference_db_diff

    # Optional smoothing in dB to avoid sharp transitions (light)
    # Apply mild moving average over iso points before interpolation
    kernel = np.ones(3) / 3.0
    pad = np.pad(relative_gains_db, (1, 1), mode='edge')
    relative_gains_db_smooth = np.convolve(pad, kernel, mode='valid')

    relative_gains_linear = 10.0 ** (relative_gains_db_smooth / 20.0)

    # Prepare interpolation nodes including DC and Nyquist with stabilized edges
    nyquist = fs / 2.0
    iso_nodes_hz = np.array([0.0] + iso_freq + [nyquist], dtype=float)

    # DC gain: copy first iso gain as a reasonable extension
    dc_gain = float(relative_gains_linear[0])
    # Nyquist: taper to 0 to reduce HF ringing
    nyq_gain = 0.0

    gain_nodes_linear = np.array([dc_gain] + list(relative_gains_linear) + [nyq_gain], dtype=float)

    # Use PCHIP (monotone) to avoid overshoot
    interpolator = PchipInterpolator(iso_nodes_hz, gain_nodes_linear, extrapolate=False)

    # Frequency grid for firwin2
    freqs = np.linspace(0.0, nyquist, numtaps, dtype=float)
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


# ===============================
# IO HELPERS
# ===============================

def save_filter_to_wav(coeff: np.ndarray,
                       filename: str,
                       fs: int,
                       channels: int = 1,
                       sample_format: str = "float32") -> None:
    """
    sample_format: 'float32' or 'pcm16'
    channels: 1 (mono) or 2 (stereo). Stereo duplicates same impulse to both channels.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if sample_format == "pcm16":
        # Normalize to 0 dBFS peak; convert to int16
        peak = np.max(np.abs(coeff)) if coeff.size > 0 else 1.0
        if peak == 0:
            peak = 1.0
        scaled = (coeff / peak * 32767.0).astype(np.int16)
        if channels == 2:
            interleaved = np.empty((scaled.size * 2,), dtype=np.int16)
            interleaved[0::2] = scaled
            interleaved[1::2] = scaled
        else:
            interleaved = scaled
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(fs)
            # Write in a single chunk for speed
            wav_file.writeframes(interleaved.tobytes())
    elif sample_format == "float32":
        coeff32 = coeff.astype(np.float32, copy=False)
        if channels == 2:
            interleaved = np.empty((coeff32.size * 2,), dtype=np.float32)
            interleaved[0::2] = coeff32
            interleaved[1::2] = coeff32
        else:
            interleaved = coeff32
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(4)
            wav_file.setframerate(fs)
            wav_file.writeframes(interleaved.tobytes())
    else:
        raise ValueError("Unsupported sample format. Use 'float32' or 'pcm16'.")


def save_response_csv(freqs: np.ndarray, gains: np.ndarray, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Frequency_Hz", "Gain_linear"])
        for fr, gn in zip(freqs, gains):
            w.writerow([float(fr), float(gn)])


# ===============================
# CLI / BATCH
# ===============================

def _progress_iter(iterable: Iterable[float], total: Optional[int], enabled: bool, desc: str = "") -> Iterable[float]:
    """
    Minimal dependency-free progress indicator.
    Prints a single-line progress with counts and percentage, updated in place.
    Falls back to no-op if not enabled or total is None.
    """
    if not enabled or total is None or total <= 0:
        for x in iterable:
            yield x
        return

    width = 30  # progress bar width
    count = 0
    for x in iterable:
        count += 1
        pct = count / total
        filled = int(pct * width)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"{desc} [{bar}] {count}/{total} ({pct*100:5.1f}%)"
        # Ensure we don't overflow terminal width too badly
        term_cols = shutil.get_terminal_size((80, 20)).columns
        if len(msg) > term_cols - 1:
            msg = msg[: term_cols - 1]
        print("\r" + msg, end="", flush=True)
        yield x
    # Ensure final 100%
    print("\r" + f"{desc} [{'#'*width}] {total}/{total} (100.0%)" + " " * max(0, term_cols - len(desc) - width - 20), flush=True)
    print()  # newline


def generate_filters(iso: str,
                     fs: int,
                     numtaps: int,
                     start_phon: float,
                     end_phon: float,
                     step_phon: float,
                     channels: int,
                     sample_format: str,
                     out_dir: str) -> None:
    curves = select_iso(iso)
    fine = create_fine_interpolated_curves(curves, ISO_FREQ, step=0.1)

    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Range handling (inclusive with rounding to one decimal)
    if step_phon <= 0:
        raise ValueError("--step-phon must be positive")

    # We allow either a single pair (start_phon,end_phon) or a range where start iterates to end.
    # To match earlier behavior, we iterate start from provided start_phon up to end_phon inclusive.
    s = round_phon_key(start_phon)
    e = round_phon_key(end_phon)
    if s > e:
        raise ValueError("start_phon must be <= end_phon")

    # Precompute how many items we will generate for progress
    total_items = int(math.floor((e - s) / step_phon + 1e-9)) + 1

    # Enable progress if stdout is a TTY
    progress_enabled = sys.stdout.isatty()

    count = 0
    def phon_generator():
        phon = s
        while phon <= e + 1e-9:
            yield phon
            phon += step_phon

    for phon in _progress_iter(phon_generator(), total_items, progress_enabled, desc="Generating"):
        phon_r = round_phon_key(phon)
        fir = design_fir_filter_from_phon_levels(phon_r, e, numtaps, fs, fine, ISO_FREQ)

        # File naming
        base = f"{phon_r:.1f}-{e:.1f}"
        wav_path = os.path.join(out_dir, f"{base}_filter.wav")
        save_filter_to_wav(fir, wav_path, fs, channels=channels, sample_format=sample_format)

        count += 1

    print(f"Generated {count} filter(s) into: {out_dir}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="FIR filter generator for equal-loudness transitions (ISO226)."
    )
    parser.add_argument("--iso", choices=["2003", "2023"], default="2023",
                        help="Select ISO dataset (default: 2023).")
    parser.add_argument("--fs", type=int, default=48000, help="Sampling rate in Hz (default: 48000).")
    parser.add_argument("--taps", type=int, default=65536, help="Number of FIR taps (default: 65536).")
    parser.add_argument("--start-phon", type=float, required=True, help="Start phon level (e.g., 60.0).")
    parser.add_argument("--end-phon", type=float, required=True, help="End phon level (e.g., 85.0).")
    parser.add_argument("--step-phon", type=float, default=0.1, help="Step for start phon (default: 0.1).")
    parser.add_argument("--channels", type=int, choices=[1, 2], default=1,
                        help="Number of channels in WAV (1=mono, 2=stereo, default: 1). Stereo duplicates the impulse.")
    parser.add_argument("--format", dest="sample_format", choices=["float32", "pcm16"], default="float32",
                        help="Output WAV sample format (default: float32).")
    parser.add_argument("--out-dir", default="output",
                        help="Output directory (default: output).")

    args = parser.parse_args(argv)

    try:
        generate_filters(
            iso=args.iso,
            fs=args.fs,
            numtaps=args.taps,
            start_phon=args.start_phon,
            end_phon=args.end_phon,
            step_phon=args.step_phon,
            channels=args.channels,
            sample_format=args.sample_format,
            out_dir=args.out_dir,
        )
    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
