import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings
import os

# Suppress FutureWarning about input name deprecation
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
from datetime import datetime
import time

torch.set_float32_matmul_precision("high")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

print("Loading model...")
model_start_time = time.time()

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

# Enable static cache but don't compile the forward pass (removes problematic torch.compile)
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256

model_load_time = time.time() - model_start_time
print(f"Model loaded in {model_load_time:.2f} seconds")

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

# Use the local MP3 file directly
# Ensure proper path formatting for Windows
audio_file_path = os.path.join(os.getcwd(), "0528.mp3")

# Check if file exists
if not os.path.exists(audio_file_path):
    print(f"Error: File not found: {audio_file_path}")
    # Look for any MP3 files in the current directory
    mp3_files = [f for f in os.listdir(os.getcwd()) if f.lower().endswith('.mp3')]
    if mp3_files:
        print(f"Found these MP3 files instead: {mp3_files}")
        audio_file_path = os.path.join(os.getcwd(), mp3_files[0])
        print(f"Using: {audio_file_path}")
    else:
        raise FileNotFoundError(f"No MP3 files found in {os.getcwd()}")
else:
    print(f"Using audio file: {audio_file_path}")

# We'll pass the file path directly to the pipeline
sample = audio_file_path

# Track processing times
processing_times = []
total_start_time = time.time()

# 2 warmup steps with timing
print("\nStarting warmup...")
warmup_times = []
for i in tqdm(range(2), desc="Warm-up steps"):
    step_start = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        result = pipe(
            sample,  # Just pass the file path directly
            chunk_length_s=120,  # Process in 30-second chunks
            generate_kwargs={
                "min_new_tokens": 256, 
                "max_new_tokens": 256, 
                "return_timestamps": True  # Required for long audio
            }
        )
    step_time = time.time() - step_start
    warmup_times.append(step_time)
    print(f"Warmup step {i+1}: {step_time:.2f} seconds")

warmup_total = sum(warmup_times)
print(f"Total warmup time: {warmup_total:.2f} seconds")

# Main processing with detailed timing
print("\nStarting main transcription...")
main_start_time = time.time()

# Create a custom progress tracker
class TranscriptionProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        self.progress_times = []
    
    def log_progress(self, percent, message=""):
        current_time = time.time()
        elapsed_from_start = current_time - self.start_time
        elapsed_from_last = current_time - self.last_progress_time
        
        self.progress_times.append({
            'percent': percent,
            'elapsed_total': elapsed_from_start,
            'elapsed_step': elapsed_from_last,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': message
        })
        
        print(f"[{percent:3d}%] {message} - Step: {elapsed_from_last:.2f}s | Total: {elapsed_from_start:.2f}s")
        self.last_progress_time = current_time

# Initialize progress tracker
tracker = TranscriptionProgressTracker()

# Simulate progress tracking during transcription
tracker.log_progress(0, "Starting transcription")

# Main transcription
transcription_start = time.time()
with sdpa_kernel(SDPBackend.MATH):
    # Pre-processing
    tracker.log_progress(10, "Pre-processing audio")
    
    # Main inference (this is where the actual work happens)
    tracker.log_progress(25, "Starting inference")
    result = pipe(
        sample,  # Just pass the file path directly
        chunk_length_s=30,  # Process in 30-second chunks
        generate_kwargs={
            "return_timestamps": True  # Required for long audio
        }
    )
    tracker.log_progress(90, "Inference completed")
    
    # Post-processing
    tracker.log_progress(95, "Post-processing")

tracker.log_progress(100, "Transcription completed")

transcription_time = time.time() - transcription_start
total_time = time.time() - total_start_time

# Add error checking for the transcription result
if not result or not (("text" in result) or ("chunks" in result)):
    print("\nError: Transcription failed to produce valid output.")
else:
    print(f"\nTranscription Result:")
    # Handle the output format from Whisper with timestamps
    if "chunks" in result and result["chunks"]:
        print("Timestamped transcription:")
        for chunk in result["chunks"]:
            try:
                print(f"[{chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s] {chunk['text']}")
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error in chunk formatting: {e}")
                print(f"Raw chunk data: {chunk}")
    elif "text" in result:
        # Handle potentially truncated text by ensuring it's properly encoded
        text = result["text"]
        try:
            # Try to re-encode the text to handle potential truncation issues
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            print(text)
        except Exception as e:
            print(f"Error formatting text: {e}")
            print("Raw text output (may contain encoding issues):")
            print(result["text"])
    else:
        print("No valid transcription text found in the result.")

# Export to text file with detailed timing information
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"whisper_transcription_{timestamp}.txt"

try:
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Whisper Large V3 Transcription Report\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Data type: {torch_dtype}\n")
        f.write(f"Audio file: {audio_file_path}\n\n")
        
        f.write(f"TIMING SUMMARY\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"Model loading time: {model_load_time:.2f} seconds\n")
        f.write(f"Warmup time: {warmup_total:.2f} seconds\n")
        f.write(f"Main transcription time: {transcription_time:.2f} seconds\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
        
        f.write(f"DETAILED PROGRESS LOG\n")
        f.write(f"-" * 25 + "\n")
        for entry in tracker.progress_times:
            f.write(f"[{entry['timestamp']}] {entry['percent']:3d}% - {entry['message']}\n")
            f.write(f"    Step time: {entry['elapsed_step']:.2f}s | Total elapsed: {entry['elapsed_total']:.2f}s\n")
        
        f.write(f"\nWARMUP DETAILS\n")
        f.write(f"-" * 15 + "\n")
        for i, wt in enumerate(warmup_times):
            f.write(f"Warmup step {i+1}: {wt:.2f} seconds\n")
        
        f.write(f"\n" + "=" * 50 + "\n")
        f.write(f"TRANSCRIPTION\n")
        f.write(f"=" * 50 + "\n\n")
        # Handle both formats of result (with or without timestamps) with error checking
        if "chunks" in result and result["chunks"]:
            # For timestamped output, format it nicely
            f.write("Timestamped transcription:\n\n")
            for chunk in result["chunks"]:
                try:
                    f.write(f"[{chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s] {chunk['text']}\n")
                except (KeyError, IndexError, TypeError) as e:
                    f.write(f"Error in chunk formatting: {e}\n")
                    f.write(f"Raw chunk data: {chunk}\n")
        elif "text" in result:
            # Handle potentially truncated text
            text = result["text"]
            try:
                # Try to re-encode the text to handle potential truncation issues
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                f.write(text)
            except Exception as e:
                f.write(f"Error formatting text: {e}\n")
                f.write("Raw text output (may contain encoding issues):\n")
                f.write(str(result["text"]))
        else:
            f.write("No valid transcription text found in the result.")
    
    print(f"\nDetailed report saved to: {filename}")
    
    # Print summary
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"Model loading: {model_load_time:.2f}s")
    print(f"Warmup: {warmup_total:.2f}s") 
    print(f"Main transcription: {transcription_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
except Exception as e:
    print(f"Error saving file: {e}")
    # Fallback to a simple filename with error handling
    try:
        with open("transcription.txt", 'w', encoding='utf-8') as f:
            if "text" in result:
                # Handle potentially truncated text
                text = result["text"].encode('utf-8', errors='replace').decode('utf-8')
                f.write(text)
            elif "chunks" in result and result["chunks"]:
                f.write("Timestamped transcription:\n\n")
                for chunk in result["chunks"]:
                    try:
                        f.write(f"[{chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s] {chunk['text']}\n")
                    except (KeyError, IndexError, TypeError):
                        f.write(f"Error in chunk data\n")
            else:
                f.write("No valid transcription data available")
        print("Transcription saved to: transcription.txt")
    except Exception as e2:
        print(f"Failed to save file: {e2}")