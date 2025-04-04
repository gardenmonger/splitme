#!/usr/bin/env python
from pitchDetect import AudioPitchDetector
from audioSplitter import AudioTransientSplitter
import GetFile as file
import GetOutput as folder
import nameThatPitch as rename

def main():
    print("Main Function")
    audio = file.get_file_path()
    output = folder.get_folder_path()
    sound = AudioPitchDetector(audio)
    splitter = AudioTransientSplitter()
    
    # Visualize the transients
    splitter.visualize_transients(audio)

    output_files = splitter.split_audio(
        audio,
        output,
        min_duration_sec=0.1,
        offset_ms=20) # Adjust detection parameters for different audio characteristics

    splitter.adjust_parameters(
        delta=0.05,            # Lower threshold for more sensitive detection
        wait=15                # Shorter wait period between transients
        )

    rename.process_wav_files(output)


if __name__ == '__main__':
    main()
