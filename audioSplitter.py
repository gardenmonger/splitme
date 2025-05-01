import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Tuple, Optional, Dict, Any


class AudioTransientSplitter:
    """
    A class that splits audio files based on detected transients with precise control
    over start and end points of each segment.
    """
    
    def __init__(self, 
                 pre_max: int = 3, 
                 post_max: int = 3,
                 pre_avg: int = 25, 
                 post_avg: int = 25,
                 wait: int = 25,
                 delta: float = 0.07,
                 sr: Optional[int] = None,
                 release_threshold: float = 0.3,
                 release_time: int = 50):
        """
        Initialize the AudioTransientSplitter with transient detection parameters.
        
        Args:
            pre_max: number of samples before a change to count as a peak
            post_max: number of samples after a change to count as a peak
            pre_avg: number of samples for calculating pre-onset average
            post_avg: number of samples for calculating post-onset average
            wait: number of samples to wait after detecting an onset
            delta: threshold difference between onset and average (lower = more sensitive)
            sr: sample rate (if None, will use the sample rate of the loaded audio)
            release_threshold: threshold for detecting transient end (0-1, fraction of peak)
            release_time: minimum time in ms for transient end detection
        """
        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.wait = wait
        self.delta = delta
        self.sr = sr
        self.release_threshold = release_threshold
        self.release_time = release_time
        self.audio_info = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa for analysis.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of audio time series (numpy.ndarray) and sample rate
        """
        # Get audio info first to access original properties
        self.audio_info = sf.info(file_path)
        
        # Load with librosa for analysis (possibly resampling)
        y, sr = librosa.load(file_path, sr=self.sr, mono=True)
        if self.sr is None:
            self.sr = sr
        return y, sr
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed audio file information including format, subtype, etc.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information
        """
        info = sf.info(file_path)
        return {
            'samplerate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'endian': info.endian,
            'duration': info.duration,
            'frames': info.frames,
            'bitrate': getattr(info, 'bitrate', None)  # Not all formats have bitrate
        }
        
    def load_audio_with_original_format(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using soundfile to preserve original format.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of audio time series (numpy.ndarray) and sample rate
        """
        # Get audio info
        info = sf.info(file_path)
        
        # Load with original sample rate
        y, sr = sf.read(file_path)
        
        # Convert to mono if necessary for analysis
        if y.ndim > 1 and y.shape[1] > 1:
            y_mono = np.mean(y, axis=1)
        else:
            y_mono = y
            
        return y_mono, sr
    
    def detect_transients(self, y: np.ndarray) -> np.ndarray:
        """
        Detect transients (onset points) in the audio signal.
        
        Args:
            y: Audio time series
            
        Returns:
            Array of transient frame indices
        """
        # Use librosa's onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, 
            sr=self.sr,
            pre_max=self.pre_max,
            post_max=self.post_max,
            pre_avg=self.pre_avg,
            post_avg=self.post_avg,
            wait=self.wait,
            delta=self.delta
        )
        
        # Convert frames to sample indices
        onset_samples = librosa.frames_to_samples(onset_frames)
        
        return onset_samples
    
    def detect_transient_segments(self, y: np.ndarray, pre_offset_samples: int = 0) -> List[Tuple[int, int]]:
        """
        Detect both the start and end points of transients in the audio signal.
        Takes pre-offset into account when calculating end points to avoid overlaps.
        
        Args:
            y: Audio time series
            pre_offset_samples: Samples to include before each transient start
            
        Returns:
            List of tuples containing (start_sample, end_sample) for each transient
        """
        # Get onset (start) points
        onset_samples = self.detect_transients(y)
        
        if len(onset_samples) == 0:
            return []
            
        # Get release (end) parameters
        release_threshold = getattr(self, 'release_threshold', 0.3)  # Default 30% of peak
        release_time_ms = getattr(self, 'release_time', 50)  # Default 50ms minimum segment length
        release_time_samples = int((release_time_ms / 1000) * self.sr)
        
        # Calculate envelope for release detection
        envelope = np.abs(y)
        
        # Find segments (start and end points)
        segments = []
        
        for i, start in enumerate(onset_samples):
            # Set minimum segment length
            min_end = start + release_time_samples
            
            # Get peak level in the early part of the segment
            if i < len(onset_samples) - 1:
                next_onset = onset_samples[i + 1]
                # Adjust end to account for pre-offset of next segment
                next_segment_start = max(0, next_onset - pre_offset_samples)
                analysis_end = min(start + release_time_samples * 2, next_onset)
            else:
                next_segment_start = len(y) - 1
                analysis_end = min(start + release_time_samples * 2, len(y) - 1)
                
            if start >= analysis_end:
                continue
                
            peak_level = np.max(envelope[start:analysis_end])
            threshold_level = peak_level * release_threshold
            
            # Find where envelope drops below threshold
            end_candidates = np.where(envelope[min_end:] < threshold_level)[0]
            
            if len(end_candidates) > 0:
                # First point below threshold after minimum length
                end = min_end + end_candidates[0]
                
                # Ensure end doesn't go beyond next segment's start (accounting for pre-offset)
                if end > next_segment_start:
                    end = next_segment_start
            else:
                # If no drop below threshold, use next segment's start or end of audio
                end = next_segment_start
                    
            segments.append((start, end))
            
        return segments
    
    def adjust_parameters(self, **kwargs):
        """
        Adjust the transient detection parameters.
        
        Args:
            **kwargs: Parameters to adjust, including:
                - pre_max: number of samples before a change to count as a peak (affects start detection)
                - post_max: number of samples after a change to count as a peak (affects end detection)
                - pre_avg: number of samples for calculating pre-onset average (start detection sensitivity)
                - post_avg: number of samples for calculating post-onset average (end detection sensitivity)
                - wait: number of samples to wait after detecting an onset
                - delta: threshold difference between onset and average (lower = more sensitive)
                - sr: sample rate
                - release_threshold: threshold for detecting transient end (0-1, default 0.3)
                - release_time: minimum time in ms for transient end detection (default 50ms)
        """
        valid_params = {
            'pre_max', 'post_max', 'pre_avg', 'post_avg', 'wait', 'delta', 'sr',
            'release_threshold', 'release_time'
        }
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
    
    def split_audio(self, 
                    file_path: str, 
                    output_dir: str, 
                    min_duration_sec: float = 0.1,
                    offset_ms: int = 0,
                    output_format: Optional[str] = None) -> List[str]:
        """
        Split audio file at detected transients.
        
        Args:
            file_path: Path to the audio file
            output_dir: Directory to save the split files
            min_duration_sec: Minimum duration in seconds for a segment to be saved
            offset_ms: Offset in milliseconds to start each segment before the detected transient
            output_format: Optional format to save files as (e.g., 'wav', 'flac').
                           If None, uses the same format as the input file.
            
        Returns:
            List of paths to the created audio segments
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio
        y, sr = self.load_audio(file_path)
        
        # Detect transients
        onsets = self.detect_transients(y)
        
        # Ensure we have the start and end points
        if len(onsets) == 0:
            return []
        
        # Add start and end points
        all_splits = np.concatenate([[0], onsets, [len(y)]])
        
        # Apply offset in samples
        offset_samples = int((offset_ms / 1000) * sr)
        
        # Determine output file extension
        if output_format:
            file_ext = f".{output_format.lower()}"
        else:
            # Use the same format as the input file
            file_ext = os.path.splitext(file_path)[1]
            if not file_ext:
                file_ext = ".wav"  # Default to wav if no extension
        
        # Get original audio format info for consistent output
        if self.audio_info is None:
            self.get_audio_info(file_path)
            
        # Split and save audio segments
        output_files = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        for i in range(len(all_splits) - 1):
            start = max(0, all_splits[i] - offset_samples)
            end = all_splits[i + 1]
            
            # Check if segment meets minimum duration
            duration = (end - start) / sr
            if duration < min_duration_sec:
                continue
                
            # Extract segment
            segment = y[start:end]
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{filename}_segment_{i:03d}{file_ext}")
            
            # Save segment with original format properties
            sf.write(
                output_file, 
                segment, 
                sr,
                subtype=self.audio_info.subtype,
                format=self.audio_info.format,
                endian=self.audio_info.endian
            )
            output_files.append(output_file)
            
        return output_files
    
    def split_in_original_format(self, 
                               file_path: str, 
                               output_dir: str, 
                               min_duration_sec: float = 0.1,
                               offset_ms: int = 0) -> List[str]:
        """
        Split audio preserving all original audio properties including channels and format.
        
        Args:
            file_path: Path to the audio file
            output_dir: Directory to save the split files
            min_duration_sec: Minimum duration in seconds for a segment to be saved
            offset_ms: Offset in milliseconds to start each segment before the detected transient
            
        Returns:
            List of paths to the created audio segments
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio and get info
        info = sf.info(file_path)
        is_multichannel = info.channels > 1
        
        # Load mono version for analysis
        y_mono, sr = self.load_audio(file_path)
        
        # Detect transients
        onsets = self.detect_transients(y_mono)
        
        # If no transients found, return empty list
        if len(onsets) == 0:
            return []
        
        # Add start and end points
        all_splits = np.concatenate([[0], onsets, [len(y_mono)]])
        
        # Apply offset in samples
        offset_samples = int((offset_ms / 1000) * sr)
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1]
        if not file_ext:
            file_ext = ".wav"  # Default to wav if no extension
        
        # Load original audio with all channels
        if is_multichannel:
            y_full, _ = sf.read(file_path)
        else:
            y_full = y_mono
        
        # Split and save audio segments
        output_files = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        for i in range(len(all_splits) - 1):
            start = max(0, all_splits[i] - offset_samples)
            end = all_splits[i + 1]
            
            # Check if segment meets minimum duration
            duration = (end - start) / sr
            if duration < min_duration_sec:
                continue
                
            # Extract segment with all original channels
            segment = y_full[start:end]
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{filename}_segment_{i:03d}{file_ext}")
            
            # Save segment with all original properties
            sf.write(
                output_file, 
                segment, 
                info.samplerate,
                subtype=info.subtype,
                format=info.format,
                endian=info.endian
            )
            output_files.append(output_file)
            
        return output_files
    
    def split_audio_by_segments(self, 
                            file_path: str, 
                            output_dir: str,
                            min_duration_sec: float = 0.1,
                            pre_offset_ms: float = 5.0) -> List[str]:
        """
        Split audio file by precisely detecting both start and end of each transient.
        Ensures segments end right before the next transient's pre-offset area.
        
        Args:
            file_path: Path to the audio file
            output_dir: Directory to save the split files
            min_duration_sec: Minimum duration in seconds for a segment to be saved
            pre_offset_ms: Milliseconds to capture before the detected transient start (pre-roll)
            
        Returns:
            List of paths to the created audio segments
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio and get info
        info = sf.info(file_path)
        is_multichannel = info.channels > 1
        
        # Load mono version for analysis
        y_mono, sr = self.load_audio(file_path)
        
        # Convert pre-offset from ms to samples
        pre_offset_samples = int((pre_offset_ms / 1000) * sr)
        
        # Detect transient segments with pre-offset consideration
        segments = self.detect_transient_segments(y_mono, pre_offset_samples)
        
        # If no segments found, return empty list
        if len(segments) == 0:
            return []
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1]
        if not file_ext:
            file_ext = ".wav"  # Default to wav if no extension
        
        # Load original audio with all channels
        if is_multichannel:
            y_full, _ = sf.read(file_path)
        else:
            y_full = y_mono
        
        # Split and save audio segments
        output_files = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        for i, (start, end) in enumerate(segments):
            # Apply pre-offset to start (ensure we don't go below 0)
            adjusted_start = max(0, start - pre_offset_samples)
            
            # Check if segment meets minimum duration
            duration = (end - adjusted_start) / sr
            if duration < min_duration_sec:
                continue
                
            # Extract segment with all original channels
            segment = y_full[adjusted_start:end]
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{filename}_transient_{i:03d}{file_ext}")
            
            # Save segment with all original properties
            sf.write(
                output_file, 
                segment, 
                info.samplerate,
                subtype=info.subtype,
                format=info.format,
                endian=info.endian
            )
            output_files.append(output_file)
            
        return output_files
    
    def visualize_transients(self, file_path: str, output_file: Optional[str] = None):
        """
        Visualize the audio waveform with detected transients (onset points only).
        
        Args:
            file_path: Path to the audio file
            output_file: Path to save the visualization (if None, will display the plot)
        """
        import matplotlib.pyplot as plt
        
        # Load audio
        y, sr = self.load_audio(file_path)
        
        # Detect transients
        onsets = self.detect_transients(y)
        
        # Create time array
        t = np.arange(len(y)) / sr
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(t, y, alpha=0.8)
        
        # Plot transients
        onset_times = onsets / sr
        plt.vlines(onset_times, -1, 1, color='r', linestyle='--', label='Transients')
        
        # Set labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform with Detected Transients')
        plt.legend()
        plt.tight_layout()
        
        # Save or display
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def visualize_transient_segments(self, file_path: str, output_file: Optional[str] = None, pre_offset_ms: float = 0):
        """
        Visualize the audio waveform with detected transient segments (start and end points).
        
        Args:
            file_path: Path to the audio file
            output_file: Path to save the visualization (if None, will display the plot)
            pre_offset_ms: Milliseconds to show before the detected transient start (for visualization)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Load audio
        y, sr = self.load_audio(file_path)
        
        # Convert pre-offset from ms to samples then to seconds for plotting
        pre_offset_samples = int((pre_offset_ms / 1000) * sr)
        pre_offset_sec = pre_offset_ms / 1000
        
        # Detect transient segments with pre-offset consideration
        segments = self.detect_transient_segments(y, pre_offset_samples)
        
        # Create time array
        t = np.arange(len(y)) / sr
        
        # Create figure
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Plot waveform
        plt.plot(t, y, color='black', alpha=0.7)
        
        # Plot segments
        colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
        
        for i, (start, end) in enumerate(segments):
            # Apply pre-offset to start (ensure we don't go below 0)
            adjusted_start = max(0, start - pre_offset_samples)
            
            start_time = start / sr
            adjusted_start_time = adjusted_start / sr
            end_time = end / sr
            duration = end_time - adjusted_start_time
            
            # Add colored rectangle for segment (including pre-offset area)
            rect = patches.Rectangle(
                (adjusted_start_time, -1), 
                duration, 
                2, 
                alpha=0.3, 
                color=colors[i],
                label=f"Transient {i}" if i == 0 else None
            )
            ax.add_patch(rect)
            
            # Add pre-offset area with different pattern if pre_offset_ms > 0
            if pre_offset_ms > 0 and adjusted_start < start:
                pre_rect = patches.Rectangle(
                    (adjusted_start_time, -1),
                    pre_offset_sec,
                    2,
                    alpha=0.15,
                    hatch='///',
                    color='gray',
                    label="Pre-roll" if i == 0 else None
                )
                ax.add_patch(pre_rect)
            
            # Add vertical lines for start and end
            plt.axvline(x=start_time, color='green', linestyle='--', alpha=0.7, 
                        label="Transient start" if i == 0 else None)
            plt.axvline(x=end_time, color='red', linestyle=':', alpha=0.7,
                        label="Transient end" if i == 0 else None)
            
            # Add text label
            plt.text(adjusted_start_time + duration/2, 0.8, f"{i}", 
                     horizontalalignment='center', fontsize=8)
        
        # Add legend
        if segments:
            plt.legend(['Waveform', 'Segment', 'Transient start', 'Transient end', 
                       'Pre-roll' if pre_offset_ms > 0 else None])
            
        # Add lines for reference
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        title = 'Audio Waveform with Detected Transient Segments'
        if pre_offset_ms > 0:
            title += f' (with {pre_offset_ms}ms pre-roll)'
        plt.title(title)
        plt.tight_layout()
        
        # Save or display
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
