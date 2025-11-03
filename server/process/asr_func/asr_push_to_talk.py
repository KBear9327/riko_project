import os
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

def transcribe_audio(model, audio_file):
    """
    Transcribes the given audio file.
    """
    segments, _ = model.transcribe(audio_file)
    transcription = " ".join([segment.text for segment in segments])
    print(f"Transcription: {transcription}")
    return transcription.strip()

def record_and_transcribe(model, output_file="recording.wav", samplerate=44100):
    """
    Simple push-to-talk recorder: record -> save -> transcribe -> return text
    """
    
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("Press ENTER to start recording...")
    input()
    
    print("🔴 Recording... Press ENTER to stop")
    
    # Record audio directly
    recording = sd.rec(int(60 * samplerate), samplerate=samplerate, channels=1, dtype='float64')
    input()  # Wait for stop
    sd.stop()
    
    print("⏹️  Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)
    
    print("🎯 Transcribing...")
    
    return transcribe_audio(model, output_file)


# Example usage
if __name__ == "__main__":
    model = WhisperModel("base.en", device="cpu", compute_type="float32")
    result = record_and_transcribe(model)
    print(f"Got: '{result}'")
    