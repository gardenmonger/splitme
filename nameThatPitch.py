import os
import argparse
from pathlib import Path
from pitchDetect import AudioPitchDetector

def process_wav_files(directory):
    """
    Find and process all .wav files in the specified directory
    
    Args:
        directory (str): Path to the directory to search for WAV files
    """
    # Convert to Path object for easier path handling
    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory")
        return
    
    # Find all .wav files (case insensitive)
    wav_files = list(dir_path.glob("**/*.wav")) + list(dir_path.glob("**/*.WAV"))
    
    # Check if any WAV files were found
    if not wav_files:
        print(f"No WAV files found in '{directory}'")
        return
    
    print(f"Found {len(wav_files)} WAV file(s) in '{directory}':")
    
    # Process each WAV file
    for i, wav_file in enumerate(wav_files, 1):
        
        print(f"{i}. {wav_file}")
        
        # Here you can add your own processing code
        # For example:
        # - Convert to another format
        # - Extract audio features
        # - Apply audio effects
        # - etc.
        
        # Example processing placeholder
        process_single_wav(wav_file)

def process_single_wav(file_path):
    """
    Process a single WAV file
    
    Args:
        file_path (Path): Path to the WAV file
    """
    # This is a placeholder for your actual processing code
    print(f"   - Processing: {file_path.name}")
    # Example: Get file size
    size_kb = file_path.stat().st_size / 1024
    print(f"   - File size: {size_kb:.2f} KB")
    
    # Add your custom processing here
    sound = AudioPitchDetector(file_path)
    sound.rename_file_with_note()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Process all WAV files in a directory")
    parser.add_argument("directory", help="Directory to search for WAV files")
    args = parser.parse_args()
    
    # Process the WAV files in the specified directory
    process_wav_files(args.directory)
