import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Tuple, Optional, Dict, Any


class AudioTransientSplitter:
    """A class that splits audio files based on detected transients with advanced tail detection."""
    
    def __init__(self, 
                 pre_max: int = 3, 
                 post_max: int = 3,
                 pre_avg: int = 25, 
                 post_avg: int = 25,
                 wait: int = 25,
                 delta: float = 0.07,
                 sr: Optional[int] = None,
                 release_threshold: float = 0.3,
                 release_time: int = 50,
                 tail_threshold: float = 0.1,
                 tail_time_ms: int = 300,
                 preserve_tail: bool = True):
        """Initialize the AudioTransientSplitter with detection parameters."""
        # Onset detection parameters
        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.wait = wait
        self.delta = delta
        self.sr = sr
        
        # Decay/tail detection parameters
        self.release_threshold = release_threshold
        self.release_time = release_time
        self.tail_threshold = tail_threshold
        self.tail_time_ms = tail_time_ms
        self.preserve_tail = preserve_tail
        
        self.audio_info = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa for analysis."""
        self.audio_info = sf.info(file_path)
        y, sr = librosa.load(file_path, sr=self.sr, mono=True)
        if self.sr is None:
            self.sr = sr
        return y, sr
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed audio file information."""
        info = sf.info(file_path)
        return {
            'samplerate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'endian': info.endian,
            'duration': info.duration,
            'frames': info.frames,
            'bitrate': getattr(info, 'bitrate', None)
        }
    
    def detect_transients(self, y: np.ndarray) -> np.ndarray:
        """Detect transient onset points in the audio signal."""
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=self.sr, pre_max=self.pre_max, post_max=self.post_max,
            pre_avg=self.pre_avg, post_avg=self.post_avg, wait=self.wait, delta=self.delta
        )
        return librosa.frames_to_samples(onset_frames)
    
    def detect_transient_segments(self, y: np.ndarray, pre_offset_samples: int = 0) -> List[Tuple[int, int]]:
        """Detect both start and end points of transients, accounting for tails and delays."""
        # Get onset points
        onset_samples = self.detect_transients(y)
        if not onset_samples.size:
            return []
        
        # Calculate smoothed envelope for improved tail detection
        frame_length, hop_length = 512, 128
        rms_envelope = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_samples(np.arange(len(rms_envelope)), hop_length=hop_length)
        
        # Interpolate to get value for each sample
        from scipy.interpolate import interp1d
        envelope_interp = interp1d(
            rms_times, rms_envelope, kind='linear', bounds_error=False,
            fill_value=(rms_envelope[0], rms_envelope[-1])
        )
        
        envelope = envelope_interp(np.arange(len(y)))
        noise_floor = np.percentile(envelope, 10)
        
        # Convert time parameters to samples
        release_time_samples = int((self.release_time / 1000) * self.sr)
        tail_time_samples = int((self.tail_time_ms / 1000) * self.sr)
        
        segments = []
        for i, start in enumerate(onset_samples):
            # Determine next segment start (for boundary checking)
            if i < len(onset_samples) - 1:
                next_onset = onset_samples[i + 1]
                next_segment_start = max(0, next_onset - pre_offset_samples)
            else:
                next_segment_start = len(y) - 1
                
            # Skip if onset is beyond signal or too close to end
            if start >= next_segment_start:
                continue
                
            # Analyze initial portion for peak detection
            analysis_window = min(start + release_time_samples * 2, next_segment_start)
            peak_level = np.max(envelope[start:analysis_window])
            
            # Set thresholds for initial release and extended tail
            release_level = peak_level * self.release_threshold
            tail_level = max(peak_level * self.tail_threshold, noise_floor * 1.5)
            
            # Find initial release point
            min_end = start + release_time_samples
            release_candidates = np.where(envelope[min_end:next_segment_start] < release_level)[0]
            
            if len(release_candidates) > 0:
                # Found initial release point
                initial_release = min_end + release_candidates[0]
                
                # Look for extended tail if enabled
                if self.preserve_tail:
                    max_tail_end = min(initial_release + tail_time_samples, next_segment_start)
                    tail_candidates = np.where(envelope[initial_release:max_tail_end] < tail_level)[0]
                    
                    if len(tail_candidates) > 0:
                        # Found tail end
                        end = initial_release + tail_candidates[0]
                    else:
                        # No clear tail end, use maximum allowed
                        end = max_tail_end
                        
                    # Check for delay repeats in latter half
                    segment_mid = start + (end - start) // 2
                    if segment_mid < end:
                        second_half = envelope[segment_mid:end]
                        secondary_peaks = np.where(second_half > peak_level * 0.4)[0]
                        if len(secondary_peaks) > 0:
                            # Found possible delay/echo, extend
                            end = next_segment_start
                else:
                    # Not preserving tails, just use initial release
                    end = initial_release
            else:
                # No clear release found, use next onset
                end = next_segment_start
            
            # Ensure end is within bounds
            end = min(end, next_segment_start)
            segments.append((start, end))
            
        return segments
    
    def adjust_parameters(self, **kwargs):
        """Adjust the transient detection parameters."""
        valid_params = {
            'pre_max', 'post_max', 'pre_avg', 'post_avg', 'wait', 'delta', 'sr',
            'release_threshold', 'release_time', 'tail_threshold', 'tail_time_ms', 'preserve_tail'
        }
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
    
    def split_audio(self, file_path: str, output_dir: str, min_duration_sec: float = 0.1,
                    offset_ms: int = 0, output_format: Optional[str] = None) -> List[str]:
        """Split audio file at detected transients."""
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        y, sr = self.load_audio(file_path)
        onsets = self.detect_transients(y)
        
        if not onsets.size:
            return []
        
        # Add start and end points
        all_splits = np.concatenate([[0], onsets, [len(y)]])
        offset_samples = int((offset_ms / 1000) * sr)
        
        # Determine output format
        file_ext = f".{output_format.lower()}" if output_format else os.path.splitext(file_path)[1]
        if not file_ext:
            file_ext = ".wav"
        
        # Get original audio info
        if self.audio_info is None:
            self.get_audio_info(file_path)
            
        # Split and save
        output_files = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        for i in range(len(all_splits) - 1):
            start = max(0, all_splits[i] - offset_samples)
            end = all_splits[i + 1]
            
            # Skip if too short
            if (end - start) / sr < min_duration_sec:
                continue
                
            segment = y[start:end]
            output_file = os.path.join(output_dir, f"{filename}_segment_{i:03d}{file_ext}")
            
            sf.write(
                output_file, segment, sr,
                subtype=self.audio_info.subtype,
                format=self.audio_info.format,
                endian=self.audio_info.endian
            )
            output_files.append(output_file)
            
        return output_files
    
    def split_audio_by_segments(self, file_path: str, output_dir: str, min_duration_sec: float = 0.1,
                               pre_offset_ms: float = 5.0, preserve_tail: Optional[bool] = None) -> List[str]:
        """Split audio with advanced transient detection including tails for delay/reverb."""
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        info = sf.info(file_path)
        y_mono, sr = self.load_audio(file_path)
        pre_offset_samples = int((pre_offset_ms / 1000) * sr)
        
        # Set temporary preserve_tail if specified
        original_preserve_tail = self.preserve_tail
        if preserve_tail is not None:
            self.preserve_tail = preserve_tail
            
        # Detect segments
        segments = self.detect_transient_segments(y_mono, pre_offset_samples)
        
        # Restore setting
        if preserve_tail is not None:
            self.preserve_tail = original_preserve_tail
            
        if not segments:
            return []
        
        # Determine file format
        file_ext = os.path.splitext(file_path)[1]
        if not file_ext:
            file_ext = ".wav"
        
        # Load full audio (including all channels)
        is_multichannel = info.channels > 1
        y_full = sf.read(file_path)[0] if is_multichannel else y_mono
        
        # Split and save
        output_files = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        for i, (start, end) in enumerate(segments):
            adjusted_start = max(0, start - pre_offset_samples)
            
            # Skip if too short
            if (end - adjusted_start) / sr < min_duration_sec:
                continue
                
            segment = y_full[adjusted_start:end]
            output_file = os.path.join(output_dir, f"{filename}_transient_{i:03d}{file_ext}")
            
            sf.write(
                output_file, segment, info.samplerate,
                subtype=info.subtype, format=info.format, endian=info.endian
            )
            output_files.append(output_file)
            
        return output_files
    
    def split_with_edited_segments(self, file_path: str, output_dir: str,
                                 edited_segments: List[Tuple[int, int]], pre_offset_ms: float = 5.0) -> List[str]:
        """Split audio using manually edited segment boundaries."""
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        info = sf.info(file_path)
        pre_offset_samples = int((pre_offset_ms / 1000) * info.samplerate)
        
        if not edited_segments:
            return []
        
        # Get file extension and load audio
        file_ext = os.path.splitext(file_path)[1] or ".wav"
        is_multichannel = info.channels > 1
        y_full = sf.read(file_path)[0] if is_multichannel else librosa.load(file_path, sr=info.samplerate, mono=True)[0]
        
        # Split and save
        output_files = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        for i, (start, end) in enumerate(edited_segments):
            adjusted_start = max(0, start - pre_offset_samples)
            segment = y_full[adjusted_start:end]
            output_file = os.path.join(output_dir, f"{filename}_segment_{i:03d}{file_ext}")
            
            sf.write(
                output_file, segment, info.samplerate,
                subtype=info.subtype, format=info.format, endian=info.endian
            )
            output_files.append(output_file)
            
        return output_files
    
    def visualize_transients(self, file_path: str, output_file: Optional[str] = None):
        """Visualize the audio waveform with detected transient onset points."""
        import matplotlib.pyplot as plt
        
        y, sr = self.load_audio(file_path)
        onsets = self.detect_transients(y)
        t = np.arange(len(y)) / sr
        
        plt.figure(figsize=(12, 6))
        plt.plot(t, y, alpha=0.8)
        plt.vlines(onsets / sr, -1, 1, color='r', linestyle='--', label='Transients')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform with Detected Transients')
        plt.legend()
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def visualize_transient_segments(self, file_path: str, output_file: Optional[str] = None, pre_offset_ms: float = 0):
        """Visualize transient segments with decay tails and pre-roll."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Load audio and detect segments
        y, sr = self.load_audio(file_path)
        pre_offset_samples = int((pre_offset_ms / 1000) * sr)
        segments = self.detect_transient_segments(y, pre_offset_samples)
        
        # Calculate envelope
        rms_envelope = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        rms_times = librosa.frames_to_samples(np.arange(len(rms_envelope)), hop_length=128) / sr
        
        # Setup plot
        t = np.arange(len(y)) / sr
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Plot waveform and envelope
        plt.plot(t, y, color='black', alpha=0.6, linewidth=1, label='Waveform')
        envelope_max = np.max(rms_envelope) if len(rms_envelope) > 0 else 1
        env_scaled = rms_envelope / envelope_max * 0.7
        plt.plot(rms_times, env_scaled, color='blue', alpha=0.6, linewidth=1.5, label='Envelope')
        
        # Plot noise floor
        if len(rms_envelope) > 0:
            noise_floor = np.percentile(rms_envelope, 10) / envelope_max * 0.7
            plt.axhline(y=noise_floor, color='green', linestyle='-.', alpha=0.4, label='Noise floor')
        
        # Plot segments
        colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
        release_time_samples = int((self.release_time / 1000) * sr)
        pre_offset_sec = pre_offset_ms / 1000
        
        for i, (start, end) in enumerate(segments):
            # Calculate times
            adjusted_start = max(0, start - pre_offset_samples)
            start_time, end_time = start / sr, end / sr
            adjusted_start_time = adjusted_start / sr
            initial_release_est = min(start + release_time_samples * 2, end)
            initial_release_time = initial_release_est / sr
            
            # Draw full segment
            rect = patches.Rectangle(
                (adjusted_start_time, -1), end_time - adjusted_start_time, 2,
                alpha=0.2, color=colors[i], label=f"Segment {i}" if i == 0 else None
            )
            ax.add_patch(rect)
            
            # Draw different parts if applicable
            if initial_release_time > start_time and initial_release_time < end_time:
                # Initial decay
                ax.add_patch(patches.Rectangle(
                    (start_time, -1), initial_release_time - start_time, 2,
                    alpha=0.3, color=colors[i], label="Initial decay" if i == 0 else None
                ))
                
                # Tail section
                if end_time > initial_release_time:
                    ax.add_patch(patches.Rectangle(
                        (initial_release_time, -1), end_time - initial_release_time, 2,
                        alpha=0.15, hatch='xxx', color=colors[i], 
                        label="Tail/Delay" if i == 0 else None
                    ))
            
            # Pre-roll area
            if pre_offset_ms > 0 and adjusted_start < start:
                ax.add_patch(patches.Rectangle(
                    (adjusted_start_time, -1), pre_offset_sec, 2,
                    alpha=0.15, hatch='///', color='gray', label="Pre-roll" if i == 0 else None
                ))
            
            # Start/end markers
            plt.axvline(x=start_time, color='green', linestyle='--', alpha=0.7, 
                        label="Transient start" if i == 0 else None)
            plt.axvline(x=end_time, color='red', linestyle=':', alpha=0.7,
                        label="Transient end" if i == 0 else None)
            
            # Segment number
            plt.text(adjusted_start_time + (end_time - adjusted_start_time)/2, 0.8, 
                     f"{i}", horizontalalignment='center', fontsize=8)
        
        # Finalize plot
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        title = 'Audio Waveform with Detected Transient Segments'
        if self.preserve_tail:
            title += ' (with tail preservation)'
        if pre_offset_ms > 0:
            title += f' (with {pre_offset_ms}ms pre-roll)'
        plt.title(title)
        plt.tight_layout()
        
        plt.figtext(0.5, 0.01, 
                   "For manual adjustment of segment boundaries, use interactive_segment_editor() method", 
                   ha="center", fontsize=10, bbox={"boxstyle": "round,pad=0.5", "alpha": 0.1})
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def interactive_segment_editor(self, file_path: str, pre_offset_ms: float = 5.0):
        """Launch interactive editor to manually adjust segment boundaries."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        import matplotlib.patches as patches
        
        # Load audio and detect segments
        y, sr = self.load_audio(file_path)
        pre_offset_samples = int((pre_offset_ms / 1000) * sr)
        segments = self.detect_transient_segments(y, pre_offset_samples)
        
        if not segments:
            print("No segments detected to edit!")
            return []
            
        # Calculate envelope for visualization
        rms_envelope = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        rms_times = librosa.frames_to_samples(np.arange(len(rms_envelope)), hop_length=128)
        envelope_max = np.max(rms_envelope) if len(rms_envelope) > 0 else 1
        env_scaled = rms_envelope / envelope_max * 0.7
        
        # Create visualization
        t = np.arange(len(y)) / sr
        
        # Setup variables
        edited_segments = segments.copy()
        current_segment = 0
        
        # Create figure
        fig, (ax1, ax_ctrl) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[4, 1])
        plt.subplots_adjust(bottom=0.2)
        
        # Function to update display
        def update_display():
            ax1.clear()
            
            # Plot waveform and envelope
            ax1.plot(t, y, color='black', alpha=0.6, linewidth=1, label='Waveform')
            ax1.plot(rms_times/sr, env_scaled, color='blue', alpha=0.6, linewidth=1.5, label='Envelope')
            
            # Plot noise floor
            if len(rms_envelope) > 0:
                noise_floor = np.percentile(rms_envelope, 10) / envelope_max * 0.7
                ax1.axhline(y=noise_floor, color='green', linestyle='-.', alpha=0.4, linewidth=1, 
                           label='Noise floor')
            
            # Plot all segments
            for i, (start, end) in enumerate(edited_segments):
                start_time = start / sr
                end_time = end / sr
                adjusted_start = max(0, start - pre_offset_samples)
                adjusted_start_time = adjusted_start / sr
                
                # Different styling for current segment
                color = 'red' if i == current_segment else 'gray'
                alpha = 0.5 if i == current_segment else 0.2
                
                # Draw segment
                rect = patches.Rectangle(
                    (adjusted_start_time, -1), end_time - adjusted_start_time, 2,
                    alpha=alpha, color=color, label=f"Segment {i}" if i == 0 else None
                )
                ax1.add_patch(rect)
                
                # Add label
                ax1.text(adjusted_start_time + (end_time - adjusted_start_time)/2, 0.8, 
                         f"{i}", horizontalalignment='center', fontsize=8)
                
                # Start/end markers
                ls = '-' if i == current_segment else ':'
                ax1.axvline(x=start_time, color='green', linestyle=ls, alpha=0.7)
                ax1.axvline(x=end_time, color='red', linestyle=ls, alpha=0.7)
            
            # Focus view on current segment
            if 0 <= current_segment < len(edited_segments):
                start, end = edited_segments[current_segment]
                start_time, end_time = start / sr, end / sr
                adjusted_start = max(0, start - pre_offset_samples)
                adjusted_start_time = adjusted_start / sr
                
                # Set view context
                context = (end_time - adjusted_start_time) * 0.5
                ax1.set_xlim(max(0, adjusted_start_time - context), end_time + context)
                
                # Update sliders without triggering callbacks
                start_slider.set_val(start_time)
                end_slider.set_val(end_time)
                
                # Set title
                ax1.set_title(f"Editing Segment {current_segment} - Drag Sliders to Adjust Boundaries")
            
            # Add axis labels and legend
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Redraw
            fig.canvas.draw_idle()
        
        # Create UI controls
        ax_start = plt.axes([0.2, 0.1, 0.65, 0.03])
        ax_end = plt.axes([0.2, 0.05, 0.65, 0.03])
        
        # Get initial values
        start, end = edited_segments[current_segment]
        start_time, end_time = start / sr, end / sr
        
        # Create sliders
        start_slider = Slider(ax_start, 'Start Time (s)', 0, t[-1], valinit=start_time)
        end_slider = Slider(ax_end, 'End Time (s)', 0, t[-1], valinit=end_time)
        
        # Create buttons
        ax_prev = plt.axes([0.2, 0.01, 0.1, 0.03])
        ax_next = plt.axes([0.35, 0.01, 0.1, 0.03])
        ax_apply = plt.axes([0.5, 0.01, 0.1, 0.03])
        ax_play = plt.axes([0.65, 0.01, 0.1, 0.03])
        ax_done = plt.axes([0.8, 0.01, 0.1, 0.03])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_apply = Button(ax_apply, 'Apply')
        btn_play = Button(ax_play, 'Play')
        btn_done = Button(ax_done, 'Done')
        
        # Button callback functions
        def on_prev(event):
            nonlocal current_segment
            if current_segment > 0:
                current_segment -= 1
                update_display()
        
        def on_next(event):
            nonlocal current_segment
            if current_segment < len(edited_segments) - 1:
                current_segment += 1
                update_display()
        
        def on_apply(event):
            if 0 <= current_segment < len(edited_segments):
                # Convert from time to samples
                new_start = int(start_slider.val * sr)
                new_end = int(end_slider.val * sr)
                
                # Validate
                if new_end <= new_start:
                    print("Error: End time must be after start time!")
                    return
                    
                # Update segment
                edited_segments[current_segment] = (new_start, new_end)
                print(f"Updated segment {current_segment}: ({new_start}, {new_end})")
                update_display()
        
        def on_play(event):
            if 0 <= current_segment < len(edited_segments):
                try:
                    import sounddevice as sd
                    
                    start, end = edited_segments[current_segment]
                    adjusted_start = max(0, start - pre_offset_samples)
                    segment = y[adjusted_start:end]
                    
                    # Play the segment
                    sd.play(segment, sr)
                except Exception as e:
                    print(f"Error playing audio: {e}")
        
        done_flag = False
        
        def on_done(event):
            nonlocal done_flag
            done_flag = True
            plt.close(fig)
        
        # Connect callbacks
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        btn_apply.on_clicked(on_apply)
        btn_play.on_clicked(on_play)
        btn_done.on_clicked(on_done)
        
        # Slider callback
        def update_sliders(val):
            if 0 <= current_segment < len(edited_segments):
                update_display()
        
        start_slider.on_changed(update_sliders)
        end_slider.on_changed(update_sliders)
        
        # Initial display
        update_display()
        plt.show()
        
        # Return edited segments or original if canceled
        return edited_segments if done_flag else segments
