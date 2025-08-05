"""Benchmark system for FIR filter performance testing."""

import platform
import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark test."""
    taps: int
    fs: int
    channels: int
    latency_ms: float


@dataclass
class SystemInfo:
    """System information for benchmark reporting."""
    cpu: str
    cores: int
    memory_gb: float
    python_version: str
    numpy_version: str
    scipy_version: str


class FIRBenchmark:
    """Performance benchmark for FIR filter processing."""
    
    VIDEO_MAX_DELAY_MS = 50.0  # 50ms max latency
    MUSIC_MAX_DELAY_MS = 200.0  # 200ms max latency
    
    TEST_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    SAMPLING_RATES = [44100, 48000, 96000]
    
    def __init__(self):
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> SystemInfo:
        """Collect basic system information."""
        try:
            import scipy
            scipy_version = scipy.__version__
        except ImportError:
            scipy_version = "not installed"
            
        return SystemInfo(
            cpu=platform.processor(),
            cores=1,  # Placeholder since we don't need actual system info
            memory_gb=1.0,  # Placeholder since we don't need actual system info
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy_version
        )
    

    
    def run_benchmark(self) -> Dict[str, Dict[int, int]]:
        """Run complete benchmark suite based on impulse response duration."""
        print("Running FIR Filter Latency Benchmark...")
        print("=" * 50)
        print("Measuring actual delay (impulse response duration)...")
        
        results: Dict[str, Dict[int, int]] = {
            'video': {},  # 50ms constraint
            'music': {}   # 200ms constraint
        }
        
        for fs in self.SAMPLING_RATES:
            print(f"\nTesting {fs/1000:.1f}kHz sampling rate...")
            
            max_video_taps = 0
            max_music_taps = 0
            
            for taps in self.TEST_SIZES:
                # Calculate actual latency in milliseconds
                latency_ms = (taps / fs) * 1000
                
                print(f"  {taps:5d} taps: {latency_ms:6.1f}ms latency")
                
                # Update maximums based on latency constraints
                if latency_ms <= self.VIDEO_MAX_DELAY_MS:
                    max_video_taps = taps
                if latency_ms <= self.MUSIC_MAX_DELAY_MS:
                    max_music_taps = taps
            
            results['video'][fs] = max_video_taps
            results['music'][fs] = max_music_taps
        
        return results
    
    def format_report(self, results: Dict[str, Dict[int, int]]) -> str:
        """Format benchmark results as human-readable report."""
        lines = []
        lines.append("FIR Filter Performance Benchmark")
        lines.append("=" * 40)
        lines.append(f"System: {self.system_info.cpu}")
        lines.append(f"Cores: {self.system_info.cores}")
        lines.append(f"Memory: {self.system_info.memory_gb:.1f}GB")
        lines.append(f"Python: {self.system_info.python_version}")
        lines.append(f"NumPy: {self.system_info.numpy_version}")
        lines.append(f"SciPy: {self.system_info.scipy_version}")
        lines.append("")
        
        lines.append("Video Playback (≤50ms delay):")
        for fs in self.SAMPLING_RATES:
            taps = results['video'][fs]
            latency_ms = (taps / fs) * 1000
            lines.append(f"- {fs/1000:.1f}kHz: Max taps = {taps:,} ({latency_ms:.1f}ms)")
        lines.append("")
        
        lines.append("Music Playback (≤200ms delay):")
        for fs in self.SAMPLING_RATES:
            taps = results['music'][fs]
            latency_ms = (taps / fs) * 1000
            lines.append(f"- {fs/1000:.1f}kHz: Max taps = {taps:,} ({latency_ms:.1f}ms)")
        lines.append("")
        
        # Recommendations based on latency
        best_video_48k = results['video'][48000]
        best_music_48k = results['music'][48000]
        
        lines.append("Recommended settings (48kHz):")
        lines.append(f"- Video: --taps {best_video_48k} --fs 48000")
        lines.append(f"- Music: --taps {best_music_48k} --fs 48000")
        
        return "\n".join(lines)
    
    def execute(self) -> int:
        """Execute benchmark and display results."""
        try:
            results = self.run_benchmark()
            report = self.format_report(results)
            print("\n" + report)
            return 0
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
            return 1
        except Exception as e:
            print(f"Benchmark failed: {e}")
            return 1