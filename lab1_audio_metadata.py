import sys
import os
from pydub import AudioSegment
from mutagen import File as MutagenFile
from mutagen import mp3, wave


def is_media_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".mp3", ".wav"]


def get_audio_info(filename: str):
    try:
        audio = AudioSegment.from_file(filename)
        duration_seconds = len(audio) / 1000.0

        print(f"File: {filename}")
        print(f"Duration: {duration_seconds:.2f} seconds")

        audio_file = MutagenFile(filename, easy=True)
        if audio_file is not None and audio_file.tags is not None:
            print("Metadata:")
            for key, value in audio_file.tags.items():
                print(f"  {key}: {value}")
        else:
            print("No metadata available")

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".mp3":
            audio_mp3 = mp3.MP3(filename)
            print("\nAdditional info (MP3):")
            print(f"  Bitrate: {audio_mp3.info.bitrate} bps")
            print(f"  Channels: {audio_mp3.info.channels}")
            print(f"  Sample rate: {audio_mp3.info.sample_rate} Hz")
            print(f"  Length: {audio_mp3.info.length:.2f} sec")

        elif ext == ".wav":
            audio_wav = wave.WAVE(filename)
            print("\nAdditional info (WAV):")
            print(f"  Channels: {audio_wav.info.channels}")
            print(f"  Sample rate: {audio_wav.info.sample_rate} Hz")
            print(f"  Bit depth: {audio_wav.info.bits_per_sample}")
            print(f"  Length: {audio_wav.info.length:.2f} sec")

    except Exception as e:
        print(f"Error while processing file: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    if not os.path.isfile(filename):
        print("File not found")
        sys.exit(1)

    if is_media_file(filename):
        get_audio_info(filename)
    else:
        print("File is not a media file of format mp3 or wav")


if __name__ == "__main__":
    main()
