import os
import numpy as np
import glob
import scipy.io.wavfile as wavfile
from scipy.fft import fft

class AudioPitchDetector:
    def __init__(self):
        # Define note frequencies (A4 = 440Hz standard)
        self.A4_freq = 440.0
        # Create a comprehensive dictionary of frequencies to notes
        self.build_note_frequencies()
        
    def build_note_frequencies(self):
        """Build a complete dictionary of note frequencies across all octaves"""
        # Notes in an octave
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Frequency of A4 is 440Hz
        # We'll calculate frequencies for other notes based on this
        self.note_frequencies = {}
        
        # Calculate for multiple octaves (0-8)
        for octave in range(9):
            for semitone, note in enumerate(notes):
                # Calculate the number of semitones from A4
                # A4 is at octave 4, position 9 (0-indexed)
                n = semitone - 9 + (octave - 4) * 12
                # Calculate frequency using the equal temperament formula
                freq = self.A4_freq * (2 ** (n / 12))
                
                note_name = f"{note}{octave}"
                self.note_frequencies[note_name] = freq
    
    def get_dominant_frequency(self, audio_data, sample_rate):
        """
        Perform FFT to find the dominant frequency in the audio
        """
        # Apply window function to reduce spectral leakage
        window = np.hamming(len(audio_data))
        windowed_data = audio_data * window
        
        # Perform FFT
        fft_result = fft(windowed_data)
        magnitude = np.abs(fft_result[:len(fft_result)//2])  # Use only first half (positive frequencies)
        
        # Create frequency bins
        freq_bins = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(audio_data)//2]
        
        # Filter out very low frequencies (below 20Hz) which are often noise
        valid_indices = freq_bins > 20
        filtered_magnitude = magnitude[valid_indices]
        filtered_freq_bins = freq_bins[valid_indices]
        
        if len(filtered_magnitude) == 0:
            return 0
            
        # Find the frequency with the highest magnitude
        max_index = np.argmax(filtered_magnitude)
        dominant_freq = filtered_freq_bins[max_index]
        
        return dominant_freq
    
    def frequency_to_note(self, frequency):
        """
        Convert a frequency to the corresponding musical note
        """
        if frequency <= 0:
            return "Unknown"
            
        # Calculate the number of semitones from A4 (440Hz)
        semitones_from_A4 = 12 * np.log2(frequency / self.A4_freq)
        # Round to the nearest semitone
        semitones_rounded = round(semitones_from_A4)
        
        # Convert semitone distance to note and octave
        note_index = (semitones_rounded + 9) % 12  # A is at index 9
        octave = 4 + (semitones_rounded + 9) // 12  # A4 is in octave 4
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = notes[note_index]
        
        # Check if octave is in valid range
        if 0 <= octave <= 8:
            return f"{note}{octave}"
        else:
            # Outside piano range, but still return a note name
            return f"{note}{octave}"
    
    def analyze_wav(self, file_path):
        """
        Analyze a WAV file to find its dominant musical note
        """
        try:
            # Read WAV file
            sample_rate, data = wavfile.read(file_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                
            # Normalize data
            data = data / np.max(np.abs(data))
            
            # Get dominant frequency
            dominant_freq = self.get_dominant_frequency(data, sample_rate)
            
            # Convert frequency to note
            note = self.frequency_to_note(dominant_freq)
            
            print(f"File: {file_path}, Dominant Frequency: {dominant_freq:.2f}Hz, Note: {note}")
            return note
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            return None
    
    def process_wav_files(self, folder_path):
        """
        Process all WAV files in the given folder and rename them with their dominant musical notes
        """
        # Get all WAV files in the folder
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        if not wav_files:
            print(f"No WAV files found in {folder_path}")
            return
        
        print(f"Found {len(wav_files)} WAV files to process")
        
        for wav_file in wav_files:
            try:
                # Get the dominant note
                note = self.analyze_wav(wav_file)
                
                if note:
                    # Create new filename with the note
                    base_path = os.path.dirname(wav_file)
                    original_filename = os.path.basename(wav_file)
                    filename_without_ext = os.path.splitext(original_filename)[0]
                    new_filename = f"{filename_without_ext}_{note}.wav"
                    new_path = os.path.join(base_path, new_filename)
                    
                    # Rename the file
                    os.rename(wav_file, new_path)
                    print(f"Renamed {original_filename} to {new_filename} (Note: {note})")
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")

