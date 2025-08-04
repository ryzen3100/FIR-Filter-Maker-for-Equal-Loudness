from typing import Iterable, Optional
import os
import csv
import shutil

import numpy as np
from scipy.io import wavfile


def save_filter_to_wav(coeff: np.ndarray,
                       filename: str,
                       fs: int,
                       channels: int = 1,
                       sample_format: str = "float32",
                       normalize_pcm16: bool = True) -> None:
    """
    sample_format: 'float32' or 'pcm16'
    channels: 1 (mono) or 2 (stereo). Stereo duplicates same impulse to both channels.
    normalize_pcm16: if True, normalize PCM16 to 0 dBFS peak; if False, clip to int16 range.
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")

    if sample_format == "pcm16":
        arr = coeff.astype(np.float64, copy=False)
        if normalize_pcm16:
            peak = np.max(np.abs(arr)) if arr.size > 0 else 1.0
            if peak == 0:
                peak = 1.0
            arr = arr / peak
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)
        if channels == 2:
            # Write as (N, 2) shaped array to indicate true stereo to wavfile.write
            stereo = np.stack((pcm, pcm), axis=-1)
            data_to_write = stereo
        else:
            data_to_write = pcm
        wavfile.write(filename, fs, data_to_write)
    elif sample_format == "float32":
        coeff32 = coeff.astype(np.float32, copy=False)
        if channels == 2:
            # Write as (N, 2) shaped array to indicate true stereo to wavfile.write
            stereo = np.stack((coeff32, coeff32), axis=-1)
            data_to_write = stereo
        else:
            data_to_write = coeff32
        wavfile.write(filename, fs, data_to_write)
    else:
        raise ValueError("Unsupported sample format. Use 'float32' or 'pcm16'.")


def save_response_csv(freqs: np.ndarray, gains: np.ndarray, filename: str) -> None:
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Frequency_Hz", "Gain_linear"])
        for fr, gn in zip(freqs, gains):
            w.writerow([float(fr), float(gn)])


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
    term_cols = shutil.get_terminal_size((80, 20)).columns
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
    print("\r" + f"{desc} [{'#'*width}] {total}/{total} (100.0%)" + " " * max(0, term_cols - len(desc) - width - 20), flush=True)
    print()
