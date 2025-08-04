#!/usr/bin/env python3
from typing import List, Literal, Optional

import os
import sys
import numpy as np
import argparse

from .iso_data import ISO_FREQ, select_iso
from .interpolation import round_phon_key, create_fine_interpolated_curves
from .design import design_fir_filter_from_phon_levels
from .io_utils import save_filter_to_wav, save_response_csv, _progress_iter


def generate_filters(iso: str,
                     fs: int,
                     numtaps: int,
                     start_phon: float,
                     end_phon: float,
                     step_phon: float,
                     channels: int,
                     sample_format: str,
                     out_dir: str,
                     smooth_db: bool = False,
                     smooth_window: int = 3,
                     dc_gain_mode: Literal["first_iso", "unity"] = "first_iso",
                     nyq_gain_db: Optional[float] = None,
                     grid_points: int = 2048,
                     export_csv: bool = False,
                     export_fir_resp: bool = False) -> None:
    curves = select_iso(iso)

    # Range handling and rounding first
    if step_phon <= 0:
        raise ValueError("--step-phon must be positive")
    s = round_phon_key(start_phon)
    e = round_phon_key(end_phon)
    if s > e:
        raise ValueError("start_phon must be <= end_phon")

    # Only build fine curves for needed range [s..e] and include end_phon
    fine = create_fine_interpolated_curves(curves, ISO_FREQ, step=0.1, needed_range=(s, e))

    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Determine deterministic step count
    n_steps = int(round((e - s) / step_phon))
    total_items = n_steps + 1

    # Enable progress if stdout is a TTY
    progress_enabled = sys.stdout.isatty()

    def phon_generator():
        for i in range(total_items):
            yield s + i * step_phon

    count = 0
    for phon in _progress_iter(phon_generator(), total_items, progress_enabled, desc="Generating"):
        phon_r = round_phon_key(phon)
        fir = design_fir_filter_from_phon_levels(
            phon_r, e, numtaps, fs, fine, ISO_FREQ,
            smooth_db=smooth_db,
            smooth_window=smooth_window,
            dc_gain_mode=dc_gain_mode,
            nyq_gain_db=nyq_gain_db,
            grid_points=grid_points
        )

        # File naming with metadata for reproducibility
        iso_tag = f"ISO{iso}"
        base = f"{iso_tag}_fs{fs}_t{numtaps}_{phon_r:.1f}-{e:.1f}"
        wav_path = os.path.join(out_dir, f"{base}_filter.wav")
        save_filter_to_wav(fir, wav_path, fs, channels=channels, sample_format=sample_format)

        if export_csv:
            # Export target grid used for firwin2
            nyquist = fs / 2.0
            freqs = np.linspace(0.0, nyquist, max(16, int(grid_points)), dtype=float)
            # Recreate gains via same parameters for exact match
            db_diff = np.array(fine[round_phon_key(e)], dtype=float) - np.array(fine[round_phon_key(phon_r)], dtype=float)
            reference_db_diff = db_diff[ISO_FREQ.index(1000)]
            relative_gains_db = db_diff - reference_db_diff
            if smooth_db and smooth_window > 1:
                kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
                pad = np.pad(relative_gains_db, (smooth_window // 2, smooth_window - 1 - smooth_window // 2), mode='edge')
                relative_gains_db = np.convolve(pad, kernel, mode='valid')
            relative_gains_linear = 10.0 ** (relative_gains_db / 20.0)
            if dc_gain_mode == "unity":
                dc_gain = 1.0
            else:
                dc_gain = float(relative_gains_linear[0])
            if nyq_gain_db is None:
                nyq_gain = 0.0
            else:
                nyq_gain = float(10.0 ** (nyq_gain_db / 20.0))
            iso_nodes_hz = np.array([0.0] + ISO_FREQ + [nyquist], dtype=float)
            gain_nodes_linear = np.array([dc_gain] + list(relative_gains_linear) + [nyq_gain], dtype=float)
            from scipy.interpolate import PchipInterpolator  # local import to avoid top-level dependency here
            interpolator = PchipInterpolator(iso_nodes_hz, gain_nodes_linear, extrapolate=False)
            gains = interpolator(freqs)
            if np.isnan(gains).any():
                gains = np.nan_to_num(gains, nan=gain_nodes_linear[-1])
            gains[0] = dc_gain
            gains[-1] = nyq_gain
            csv_path = os.path.join(out_dir, f"{base}_response.csv")
            save_response_csv(freqs, gains, csv_path)

        if export_fir_resp:
            # Optional: export magnitude response of the designed FIR
            from scipy import signal
            w, h = signal.freqz(fir, worN=4096, fs=fs)
            mag = np.abs(h)
            csv_path2 = os.path.join(out_dir, f"{base}_fir_mag.csv")
            save_response_csv(w, mag, csv_path2)

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

    # New options
    parser.add_argument("--smooth-db", action="store_true",
                        help="Enable mild smoothing in dB across ISO points.")
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="Smoothing window length (odd integer, default: 3).")
    parser.add_argument("--dc-gain-mode", choices=["first_iso", "unity"], default="first_iso",
                        help="DC gain choice: extend first ISO gain or force unity (default: first_iso).")
    parser.add_argument("--nyq-gain-db", type=float, default=None,
                        help="Nyquist gain in dB. Default None tapers to 0 linear.")
    parser.add_argument("--grid-points", type=int, default=2048,
                        help="Number of frequency grid points for target response (default: 2048).")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export the PCHIP target grid (freqs,gains) for each filter.")
    parser.add_argument("--export-fir-resp", action="store_true",
                        help="Export the designed FIR magnitude response as CSV.")

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
            smooth_db=args.smooth_db,
            smooth_window=args.smooth_window,
            dc_gain_mode=args.dc_gain_mode, 
            nyq_gain_db=args.nyq_gain_db,
            grid_points=args.grid_points,
            export_csv=args.export_csv,
            export_fir_resp=args.export_fir_resp,
        )
    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 1
    return 0
