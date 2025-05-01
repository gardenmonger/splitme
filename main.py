#!/usr/bin/env python
from pitchDetect import AudioPitchDetector
from audioSplitter import AudioTransientSplitter
import GetFile as file
import GetOutput as folder

def main():
    print("Main Function")
    audio = file.get_file_path()
    output = folder.get_folder_path()
    splitter = AudioTransientSplitter(
        delta=0.05,           # Lower threshold for more sensitive detection
        wait=45,              # Wait time between onsets
    
        # End detection (release) parameters
        release_threshold=0.01,  # Release threshold (40% of peak)
        release_time=1000         # Minimum segment duration (30ms)
    )
    
    # Visualize the transients
    splitter.visualize_transient_segments(audio)

    splitter.adjust_parameters(
        # For onset (start) detection:
        delta=0.08,              # More sensitive onset detection
        release_threshold=0.2,  # Lower threshold = longer decay tails
        # release_time=50,          # Slightly longer minimum duration
        wait=30,                 # Allow closer transients
        
        # For release (end) detection:
        release_time=1500          # Longer minimum segment length (40ms)
    )

    #  # Visualize the transients re-adjusted
    # splitter.visualize_transient_segments(audio)


    output_files = splitter.split_audio_by_segments(
        audio,
        output,
        min_duration_sec=0.1,
        pre_offset_ms=20        # 12ms pre-roll
        # offset_ms=10 # Adjust detection parameters for different audio characteristics
    ) 

    rename = AudioPitchDetector()

    rename.process_wav_files(output)



    

if __name__ == '__main__':
    main()

   