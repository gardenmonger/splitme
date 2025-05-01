import os
import numpy as np
import wave
import glob
from scipy.fft import fft
import scipy.io.wavfile as wavfile

class AudioPitchDetector:
    def __init__(self):
        # Dictionary mapping frequencies to musical notes
        # Frequencies based on A4 = 440Hz
        self.notes = {
            16.35: "C0",
            17.32: "C#0",
            18.35: "D0",
            19.45: "D#0",
            20.60: "E0",
            21.83: "F0",
            23.12: "F#0",
            24.50: "G0",
            25.96: "G#0",
            27.50: "A0",
            29.14: "A#0",
            30.87: "B0"
        }
        
        # Populate all octaves
        for i in range(1, 9):
            for base_freq, note in list(self.notes.items()):
                if note.endswith(str(i-1)):
                    new_note = note[:-1] + str(i)
                    self.notes[base_freq * (2**i)] = new_note
    
    def analyze_wav(self, file_path):
        """
        Analyze a single WAV file and return the dominant musical note
        """
        try:
            # Read WAV file
            sample_rate, data = wavfile.read(file_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Apply window function to reduce spectral leakage
            window = np.hamming(len(data))
            data = data * window
            
            # Perform FFT and get magnitude spectrum
            fft_result = fft(data)
            magnitude = np.abs(fft_result)
            
            # Get frequency bins
            freq_bins = np.fft.fftfreq(len(magnitude), 1/sample_rate)
            
            # Consider only positive frequencies
            positive_freq_indices = np.where(freq_bins > 0)
            magnitude = magnitude[positive_freq_indices]
            freq_bins = freq_bins[positive_freq_indices]
            
            # Find the frequency with the highest magnitude
            peak_index = np.argmax(magnitude)
            dominant_frequency = freq_bins[peak_index]
            
            # Find the closest musical note
            closest_note = self.find_closest_note(dominant_frequency)
            
            return closest_note
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            return None
    
    def find_closest_note(self, frequency):
        """
        Find the closest musical note to the given frequency
        """
        # Convert all notes to a list for easier processing
        notes_list = list(self.notes.items())
        notes_list.sort()  # Sort by frequency
        
        # Handle frequencies beyond our range
        if frequency < notes_list[0][0]:
            return notes_list[0][1]  # Return the lowest note
        if frequency > notes_list[-1][0]:
            return notes_list[-1][1]  # Return the highest note
        
        # Find the two closest notes
        for i in range(len(notes_list) - 1):
            if notes_list[i][0] <= frequency < notes_list[i+1][0]:
                # Choose the closer note
                if abs(frequency - notes_list[i][0]) < abs(frequency - notes_list[i+1][0]):
                    return notes_list[i][1]
                else:
                    return notes_list[i+1][1]
                
        # Fallback
        return "Unknown"
    
    def process_wav_files(self, folder_path):
        """
        Process all WAV files in the given folder and rename them with their musical notes
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

