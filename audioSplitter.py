import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Tuple, Optional, Dict, Any

class AudioTransientSplitter:
    """
    A class that splits audio files based on detected transients.
    """
    
    def __init__(self, 
                 pre_max: int = 3, 
                 post_max: int = 3,
                 pre_avg: int = 25, 
                 post_avg: int = 25,
                 wait: int = 25,
                 delta: float = 0.07,
                 sr: Optional[int] = None):
        """
        Initialize the AudioTransientSplitter with transient detection parameters.
        
        Args:
            pre_max: number of samples before a change to count as a peak
            post_max: number of samples after a change to count as a peak
            pre_avg: number of samples for calculating pre-onset average
            post_avg: number of samples for calculating post-onset average
            wait: number of samples to wait after detecting an onset
            delta: threshold difference between onset and average
            sr: sample rate (if None, will use the sample rate of the loaded audio)
        """
        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.wait = wait
        self.delta = delta
        self.sr = sr
        self.audio_info = None
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
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
    
    def detect_transients(self, y: np.ndarray) -> np.ndarray:
        """
        Detect transients in the audio signal.
        
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
    
    def adjust_parameters(self, **kwargs):
        """
        Adjust the transient detection parameters.
        
        Args:
            **kwargs: Parameters to adjust (pre_max, post_max, pre_avg, post_avg, wait, delta)
        """
        valid_params = {'pre_max', 'post_max', 'pre_avg', 'post_avg', 'wait', 'delta', 'sr'}
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
                
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
                
    def visualize_transients(self, file_path: str, output_file: Optional[str] = None):
        """
        Visualize the audio waveform with detected transients.
        
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