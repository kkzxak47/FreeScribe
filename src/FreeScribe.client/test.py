import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Load the Whisper model (choose 'base', 'small', 'medium', or 'large')
model = WhisperModel("medium", device="cpu", compute_type="int8")  # Use GPU if available by setting device="cuda"

# Set parameters
sample_rate = 16000  # Whisper expects 16kHz audio
language = "fr"  # Input language: French
target_language = "en"  # Translation target language: English

audio_queue = queue.Queue()

def select_microphone():
    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']}")
    
    device_id = int(input("Select the input device by entering the corresponding number: "))
    return device_id

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def transcribe_audio():
    print("Starting transcription...")
    audio_buffer = np.zeros(0, dtype=np.float32)

    while True:
        # Retrieve audio data from the queue
        try:
            data = audio_queue.get()
            audio_buffer = np.concatenate((audio_buffer, data[:, 0]))  # Use the first channel
        except queue.Empty:
            continue

        # Process in chunks of 30 seconds (sample_rate * 30 samples)
        if len(audio_buffer) >= sample_rate * 30:
            audio_chunk = audio_buffer[: sample_rate * 30]
            audio_buffer = audio_buffer[sample_rate * 30 :]

            # Transcribe and translate
            segments, _ = model.transcribe(audio_chunk, language=language, task="translate", beam_size=5)
            for segment in segments:
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

if __name__ == "__main__":
    try:
        # Select microphone
        device_id = select_microphone()

        # Start audio stream
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, callback=audio_callback, device=device_id):
            transcribe_audio()
    except KeyboardInterrupt:
        print("Transcription stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
