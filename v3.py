import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
from pathlib import Path
import datetime
import time
from tqdm import tqdm


class ProgressCallback:
    def __init__(self, total_duration):
        self.total_duration = total_duration
        self.processed_duration = 0
        self.start_time = time.time()
        
    def __call__(self, step_info=None):
        # This is a simple callback - actual implementation depends on the pipeline
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Estimate progress (this is approximate)
        if elapsed > 0:
            estimated_total_time = elapsed * (self.total_duration / max(self.processed_duration, 1))
            progress = min(self.processed_duration / self.total_duration * 100, 100)
            print(f"\rProgress: {progress:.1f}% | Elapsed: {elapsed:.1f}s", end="", flush=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

print("Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

print(f"Device set to use {device}")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    chunk_length_s=30,
    device=device,
    return_timestamps=True
)

# Load the local Thai audio file
audio_file = "0528.MP3"
print(f"Loading audio file: {audio_file}")

# Load the audio file using librosa
audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
total_duration = len(audio_array) / sampling_rate

print(f"Audio duration: {total_duration:.2f} seconds")
print(f"Processing {total_duration/30:.1f} chunks of 30 seconds each")

# Create the sample dictionary in the format expected by the pipeline
sample = {
    "array": audio_array,
    "sampling_rate": sampling_rate
}

# Progress tracking setup
chunk_size = 30  # seconds
total_chunks = int(np.ceil(total_duration / chunk_size))
print(f"\nStarting transcription...")
print("=" * 50)

# Custom progress tracking by processing the pipeline
start_time = time.time()

# Unfortunately, the Transformers pipeline doesn't expose internal progress directly
# So we'll create a progress bar and update it based on time estimates
with tqdm(total=100, desc="Transcribing", unit="%", ncols=70) as pbar:
    def update_progress():
        elapsed = time.time() - start_time
        # Rough estimate: assume processing speed is somewhat consistent
        if elapsed > 5:  # After 5 seconds, start estimating
            estimated_total = elapsed * (total_duration / 30)  # rough estimate
            progress = min((elapsed / estimated_total) * 100, 95)  # cap at 95% until done
            pbar.n = int(progress)
            pbar.last_print_n = int(progress)
            pbar.refresh()
    
    # Start a simple progress updater (this runs in background conceptually)
    import threading
    
    progress_thread = None
    stop_progress = False
    
    def progress_updater():
        while not stop_progress:
            update_progress()
            time.sleep(2)  # Update every 2 seconds
    
    progress_thread = threading.Thread(target=progress_updater)
    progress_thread.daemon = True
    progress_thread.start()
    
    # Run the actual transcription
    result = pipe(sample, generate_kwargs={"language": "thai"})
    
    # Stop progress tracking
    stop_progress = True
    pbar.n = 100
    pbar.last_print_n = 100
    pbar.refresh()

end_time = time.time()
processing_time = end_time - start_time

print(f"\n\nTranscription completed!")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Audio duration: {total_duration:.2f} seconds")
print(f"Processing speed: {total_duration/processing_time:.2f}x realtime")

# Create output filename based on input file
input_path = Path(audio_file)
output_file = input_path.stem + "_transcription.txt"

# Write results to text file
print(f"\nSaving transcription to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    # Write header with metadata
    f.write(f"Transcription of: {audio_file}\n")
    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {model_id}\n")
    f.write(f"Audio duration: {total_duration:.2f} seconds\n")
    f.write(f"Processing time: {processing_time:.2f} seconds\n")
    f.write(f"Processing speed: {total_duration/processing_time:.2f}x realtime\n")
    f.write("-" * 50 + "\n\n")
    
    if "chunks" in result:
        f.write("Transcription with timestamps:\n\n")
        for chunk in result["chunks"]:
            timestamp = chunk['timestamp']
            if timestamp:
                start_time = timestamp[0] if timestamp[0] is not None else 0
                end_time = timestamp[1] if timestamp[1] is not None else "end"
                f.write(f"[{start_time:.2f} - {end_time}]: {chunk['text']}\n")
            else:
                f.write(f"{chunk['text']}\n")
        
        # Also write full text without timestamps
        f.write("\n" + "-" * 50 + "\n")
        f.write("Full transcription (no timestamps):\n\n")
        full_text = " ".join([chunk['text'].strip() for chunk in result["chunks"]])
        f.write(full_text)
    else:
        f.write("Full transcription:\n\n")
        f.write(result["text"])

print("Transcription with timestamps:")
if "chunks" in result:
    for chunk in result["chunks"]:
        print(f"{chunk['timestamp']}: {chunk['text']}")
else:
    print(result["text"])

print(f"\nTranscription saved to: {output_file}")
print("Done!")