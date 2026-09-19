import threading
import queue
import time
import subprocess  # Used to launch another process
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
TARGET_PHRASE = "test test"  # <-- CHANGE THIS to your specific phrase (lowercase)
RATE = 16000
CHUNK = 1024
SILENCE_LIMIT = 1.2         # Seconds of silence before finalizing a phrase
LOUDNESS_THRESHOLD = 400    # Mic sensitivity (lower = more sensitive)

print("Loading Whisper model...")
model = WhisperModel("base.en", device="cpu", compute_type="float32")
print("Model loaded!")

audio_queue = queue.Queue()
is_running = True

def run_external_process():
    """
    This function handles launching your other process so it doesn't 
    freeze or lag the audio listener thread.
    """
    print(f"\n[!!!] TARGET PHRASE DETECTED: Running external process...")
    
    # Example 1: Run an external python script or system command
    # subprocess.Popen(["python", "my_other_script.py"])
    
    # Example 2: Just run some internal function
    # your_custom_function()
    
    time.sleep(2) # Dummy delay representing work
    print("[System] External process initialized.\n")

def audio_stream_worker():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while is_running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
        except Exception as e:
            print(f"Stream error: {e}")
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcription_worker():
    global is_running
    speaking_buffer = []
    silent_chunks_count = 0
    max_silent_chunks = int((RATE / CHUNK) * SILENCE_LIMIT)
    
    print(f"\n--- Standing by. Say '{TARGET_PHRASE}' to trigger the process ---")
    
    while is_running:
        if audio_queue.empty():
            time.sleep(0.01)
            continue
            
        raw_chunk = audio_queue.get()
        chunk_array = np.frombuffer(raw_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(chunk_array.astype(np.float32)**2))
        
        if rms > LOUDNESS_THRESHOLD:
            float_chunk = chunk_array.astype(np.float32) / 32768.0
            speaking_buffer.append(float_chunk)
            silent_chunks_count = 0 
        else:
            if len(speaking_buffer) > 0:
                float_chunk = chunk_array.astype(np.float32) / 32768.0
                speaking_buffer.append(float_chunk)
                silent_chunks_count += 1
                
                if silent_chunks_count > max_silent_chunks:
                    full_audio = np.concatenate(speaking_buffer)
                    speaking_buffer = []
                    silent_chunks_count = 0
                    
                    # Process text
                    segments, _ = model.transcribe(full_audio, beam_size=3)
                    for segment in segments:
                        text = segment.text.strip().lower()
                        # Clean up basic punctuation Whisper might add
                        clean_text = text.replace(".", "").replace(",", "").replace("!", "").strip()
                        
                        if clean_text:
                            print(f"Heard: \"{clean_text}\"")
                            
                            # Check if your specific phrase is inside what was said
                            if TARGET_PHRASE in clean_text:
                                # Fire the target process on a separate thread so listening doesn't freeze
                                proc_thread = threading.Thread(target=run_external_process)
                                proc_thread.start()

if __name__ == "__main__":
    t1 = threading.Thread(target=audio_stream_worker, daemon=True)
    t2 = threading.Thread(target=transcription_worker, daemon=True)
    t1.start()
    t2.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down custom trigger stream...")
        is_running = False
