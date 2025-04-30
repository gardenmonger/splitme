#!/usr/bin/env python

import librosa
import os
import numpy as np
import math

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
        Convert a frequency in Hz to the closest musical note.
        
        Args:
            frequency (float): The frequency in Hz to convert
            
        Returns:
            tuple: (note_name, octave, cents_deviation)
                - note_name: The name of the note (C, C#, D, etc.)
                - octave: The octave number
                - cents_deviation: How many cents sharp/flat the frequency is from the exact note
        """
        # A4 = 440 Hz is our reference
        A4 = 440.0
        # A4 = 442.0
        
        # Notes in an octave
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Check for extremely low frequencies
        if frequency <= 0:
            return None, None, None
        
        # Calculate the number of half steps away from A4
        # 12 * log2(f/440) gives the number of half steps
        half_steps = 12 * math.log2(frequency / A4)
        
        # Round to the closest whole number to get the closest note
        closest_half_step = round(half_steps)
        
        # Calculate how many cents we are off from the exact note
        # 100 cents in a half step, positive means sharp, negative means flat
        cents_deviation = 100 * (half_steps - closest_half_step)
        
        # A4 is note 'A' at octave 4, which is the 9th position (index 9) from C0
        # So A4 is at position 9 + 4*12 = 57 half steps from C0
        # C0 is the reference point (considered to be the lowest note)
        
        # Calculate the absolute position from C0
        absolute_position = 57 + closest_half_step  # 57 is the position of A4
        
        # Determine the octave
        octave = (absolute_position // 12) - 1  # Integer division
        
        # Determine the note name (index in the notes list)
        note_index = absolute_position % 12
        note_index = (note_index + 3) % 12  # Adjust because our array starts with C but we calculated from A
        
        # Get the note name
        note_name = notes[note_index]
        
        print(f"octave: {octave}{cents_deviation}")

        fnote = f"{note_name}_{octave}"
        print(fnote)

        return fnote

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
