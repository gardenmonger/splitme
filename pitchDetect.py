#!/usr/bin/env python

import librosa
import os
import numpy as np

class AudioPitchDetector:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audio, self.sr = librosa.load(audio_path, sr=None)  # Load audio with its original sample rate

    def audio_sampleRate(self):
        sr = self.sr
        return sr

    def detect_pitch(self):
        """
        Detect the pitch of the audio and return it as a musical note.
        """
        # Perform pitch detection using librosa's pyin algorithm
        # This will give us an estimate of the pitch in Hz
        pitches, magnitudes = librosa.core.piptrack(y=self.audio, sr=self.sr)

        # Handle the case where magnitudes are empty or there's an issue with pitch detection
        if pitches.size == 0 or magnitudes.size == 0:
            print("Error: Pitch or magnitude data is empty.")
            return None

        # Get the index of the peak in the pitch array (we need to ensure we don't go out of bounds)
        # We can limit the index search to the last valid index to prevent out-of-bounds errors
        index_of_peak = magnitudes[:, :].argmax(axis=0)

        # Handle cases where magnitudes might not have a clear peak
        # We limit the index_of_peak to avoid IndexErrors at the edges
        pitch_hz = None
        for i in range(len(index_of_peak)):
            try:
                pitch_hz = pitches[index_of_peak[i], i]
            except IndexError:
                # Handle index out of bounds by skipping problematic points
                continue
            
            # If a valid pitch is found, stop the loop
            if pitch_hz is not None:
                break

        if pitch_hz is None:
            print("Error: No valid pitch found.")
            return None
        
        # Convert Hz to musical note
        note = self.hz_to_note(pitch_hz)
        return note

    def hz_to_note(self, frequency):
        """
        Convert frequency in Hz to the closest musical note (A, B, C, D, E, F, G).
        """
        # Define note frequencies (for standard tuning, A4 = 440 Hz)
        note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
            'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
            'A#': 466.16, 'B': 493.88
        }

        # Calculate the closest note
        diff = np.inf
        closest_note = None
        for note, note_freq in note_frequencies.items():
            note_diff = abs(note_freq - frequency)
            if note_diff < diff:
                diff = note_diff
                closest_note = note
        
        return closest_note

    def rename_file_with_note(self):
        """
        Rename the audio file by appending the detected note to the original file name.
        """
        # Detect the pitch
        note = self.detect_pitch()
        
        # If no note could be detected, do not rename the file
        if note is None:
            print("Error: Could not detect pitch, file will not be renamed.")
            return None

        # Get the file name and extension
        base_name, ext = os.path.splitext(os.path.basename(self.audio_path))

        # Create the new file name
        new_name = f"{base_name}_{note}{ext}"

        # Get the directory where the file is located
        directory = os.path.dirname(self.audio_path)

        # Construct the full path to rename the file
        new_path = os.path.join(directory, new_name)

        # Rename the file
        os.rename(self.audio_path, new_path)

        print(f"File renamed to: {new_path}")
        return new_path

# Example usage
#audio_path = 'piano.wav'  # Replace with your audio file path
#pitch_detector = AudioPitchDetector(audio_path)
#pitch_detector.rename_file_with_note()


#1 grabs the audio file


#2 detects the pitch of audio and returns pitch value as a note like "A,G,C,F" in string format


#3 renames the file with its original name but append the note to the name of audio file, example "piano.wav" becomes "piano_A_.wav"
