import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import time
from transformers import pipeline
from io import BytesIO
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import math

# Define the models
MODELS = {
    "Whisper (English)": "openai/whisper-small.en",
    "Whisper (Multilingual)": "openai/whisper-small",
    "Facebook Wav2Vec2": "facebook/wav2vec2-large-960h",
    "Google Wav2Vec2": "google/wav2vec2-large-xlsr-53",
    "Whisper (Thai)": "openai/whisper-large",
    "Whisper Large V3": "openai/whisper-large-v3"
}

def split_audio_into_chunks(audio_data, sr, chunk_duration=10):
    """Split audio into chunks of specified duration (in seconds)"""
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        start_time = i / sr
        end_time = min((i + chunk_samples) / sr, len(audio_data) / sr)
        chunks.append({
            'data': chunk,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        })
    
    return chunks

def process_chunk(chunk_data, model, chunk_info, chunk_index):
    """Process a single audio chunk"""
    try:
        # Create temporary file for this chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, chunk_data, 16000, format='WAV')
            
            # Process with model
            result = model(tmp_file.name, return_timestamps=True)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return {
                'chunk_index': chunk_index,
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'text': result['text'],
                'success': True
            }
    except Exception as e:
        return {
            'chunk_index': chunk_index,
            'start_time': chunk_info['start_time'],  
            'end_time': chunk_info['end_time'],
            'text': f"Error processing chunk: {str(e)}",
            'success': False
        }

def convert_audio_to_text_with_chunks(audio_input, model_name, sr=None, is_file_path=True):
    """Convert audio to text with detailed chunk-by-chunk progress tracking"""
    
    # Create placeholders for status updates
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    chunk_status_placeholder = st.empty()
    results_placeholder = st.empty()
    
    try:
        # Step 1: Initialize model
        status_placeholder.info("🔄 Initializing model...")
        progress_bar.progress(5)
        time.sleep(0.5)
        
        model = pipeline("automatic-speech-recognition", model=MODELS[model_name])
        
        # Step 2: Load audio if it's a file path
        status_placeholder.info("📁 Loading audio file...")
        progress_bar.progress(10)
        
        if is_file_path:
            audio_data, sample_rate = librosa.load(audio_input, sr=16000)
        else:
            audio_data = audio_input
            sample_rate = sr if sr else 16000
            
        total_duration = len(audio_data) / sample_rate
        
        # Step 3: Split audio into chunks
        status_placeholder.info("✂️ Splitting audio into chunks...")
        progress_bar.progress(15)
        
        chunk_duration = 10  # 10 seconds per chunk
        chunks = split_audio_into_chunks(audio_data, sample_rate, chunk_duration)
        total_chunks = len(chunks)
        
        st.info(f"📊 **Processing Details:**\n- Total Duration: {total_duration:.1f} seconds\n- Number of Chunks: {total_chunks}\n- Chunk Size: {chunk_duration} seconds each")
        
        # Step 4: Process chunks with detailed progress
        status_placeholder.info("🗣️ Converting speech to text...")
        
        all_results = []
        processed_chunks = 0
        start_time = time.time()
        
        # Create a container for live results
        live_results_container = st.container()
        
        for i, chunk in enumerate(chunks):
            # Update main progress
            main_progress = 15 + (70 * (i + 1) / total_chunks)
            progress_bar.progress(int(main_progress))
            
            # Show current chunk being processed
            chunk_status_placeholder.info(
                f"🎵 Processing chunk {i+1}/{total_chunks} "
                f"({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)"
            )
            
            # Process the chunk
            chunk_start_time = time.time()
            result = process_chunk(chunk['data'], model, chunk, i)
            chunk_end_time = time.time()
            
            chunk_processing_time = chunk_end_time - chunk_start_time
            processed_chunks += 1
            
            # Add result to collection
            all_results.append(result)
            
            # Update live results display
            with live_results_container:
                # Show processing stats
                elapsed_time = time.time() - start_time
                avg_time_per_chunk = elapsed_time / processed_chunks
                estimated_remaining = avg_time_per_chunk * (total_chunks - processed_chunks)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Processed", f"{processed_chunks}/{total_chunks}")
                with col2:
                    st.metric("Elapsed", f"{elapsed_time:.1f}s")
                with col3:
                    st.metric("Remaining", f"{estimated_remaining:.1f}s")
                with col4:
                    st.metric("Speed", f"{chunk_processing_time:.1f}s/chunk")
                
                # Show latest processed text
                if result['success']:
                    st.success(f"✅ Chunk {i+1} ({chunk['start_time']:.1f}s-{chunk['end_time']:.1f}s): \"{result['text'][:100]}{'...' if len(result['text']) > 100 else ''}\"")
                else:
                    st.error(f"❌ Chunk {i+1} failed: {result['text']}")
            
            # Small delay to show progress
            time.sleep(0.1)
        
        # Step 5: Combine results
        status_placeholder.info("🔄 Combining results...")
        progress_bar.progress(90)
        
        # Combine all successful chunks
        combined_text = ""
        successful_chunks = [r for r in all_results if r['success']]
        
        for result in sorted(successful_chunks, key=lambda x: x['start_time']):
            combined_text += result['text'] + " "
        
        combined_text = combined_text.strip()
        
        # Step 6: Complete
        end_time = time.time()
        total_conversion_time = end_time - start_time
        
        status_placeholder.success("✅ Conversion completed successfully!")
        chunk_status_placeholder.success(f"🎉 All {total_chunks} chunks processed!")
        progress_bar.progress(100)
        
        return {
            'text': combined_text,
            'chunks': all_results,
            'total_time': total_conversion_time,
            'total_chunks': total_chunks,
            'successful_chunks': len(successful_chunks),
            'failed_chunks': total_chunks - len(successful_chunks)
        }, total_conversion_time
        
    except Exception as e:
        status_placeholder.error(f"❌ Error during conversion: {str(e)}")
        progress_bar.progress(0)
        return None, 0

# Function for simple conversion (fallback)
def convert_audio_to_text_simple(audio_input, model_name, is_file_path=True):
    """Simple conversion without chunking (fallback method)"""
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_placeholder.info("🔄 Initializing model...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        model = pipeline("automatic-speech-recognition", model=MODELS[model_name])
        
        status_placeholder.info("✅ Model loaded successfully!")
        progress_bar.progress(30)
        time.sleep(0.5)
        
        status_placeholder.info("🗣️ Converting speech to text...")
        progress_bar.progress(70)
        
        start_time = time.time()
        
        if is_file_path:
            result = model(audio_input, return_timestamps=True)
        else:
            result = model(audio_input)
        
        end_time = time.time()
        conversion_time = end_time - start_time
        
        status_placeholder.success("✅ Conversion completed successfully!")
        progress_bar.progress(100)
        
        return result, conversion_time
        
    except Exception as e:
        status_placeholder.error(f"❌ Error during conversion: {str(e)}")
        progress_bar.progress(0)
        return None, 0

# App UI
st.title("🎙️ Audio to Text Conversion")
st.subheader("Select language and model")

# Language selection
language = st.selectbox("Choose Language", options=["English", "Thai"])

# Model selection
model_choice = st.selectbox("Choose a Model", options=list(MODELS.keys()))

# Processing method selection
processing_method = st.radio(
    "Choose Processing Method:",
    options=["Detailed Progress (Chunk-based)", "Simple Progress"],
    help="Detailed Progress shows second-by-second conversion progress but takes longer. Simple Progress is faster but shows less detail."
)

# Audio input options
st.subheader("Record or Upload your audio")
audio_option = st.radio("Choose an option:", ('Record Audio', 'Upload Audio'))

audio_data = None

# Queue to store recorded audio frames
audio_queue = queue.Queue()

# WebRTC Audio Recorder
def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()
    audio_queue.put(audio)
    return frame

# Option 1: Record audio via browser using WebRTC
if audio_option == 'Record Audio':
    st.write("Click the button to start/stop recording.")

    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun1.l.google.com:19302"]}, 
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        }
    )

    webrtc_ctx = webrtc_streamer(
        key="audio-stream",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_frame_callback,
    )

    if webrtc_ctx.state.playing:
        st.write("🔴 Recording in progress...")
        
    elif not webrtc_ctx.state.playing and not audio_queue.empty():
        st.write("🎵 Processing recorded audio...")
        
        recorded_audio = []
        while not audio_queue.empty():
            recorded_audio.append(audio_queue.get())

        if recorded_audio:
            audio_data = np.concatenate(recorded_audio, axis=0)
            sr = 16000

            audio_size = len(audio_data) * 2
            duration = len(audio_data) / sr

            st.info(f"""
            📊 **Audio Properties:**
            - Size: {audio_size:,} bytes
            - Sample Rate: {sr:,} Hz
            - Duration: {duration:.2f} seconds
            """)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sr, format='WAV')
                tmp_file_path = tmp_file.name

            st.subheader("🔄 Audio Conversion Status")
            
            if processing_method == "Detailed Progress (Chunk-based)":
                result, conversion_time = convert_audio_to_text_with_chunks(tmp_file_path, model_choice, sr, is_file_path=True)
            else:
                result, conversion_time = convert_audio_to_text_simple(tmp_file_path, model_choice, is_file_path=True)

            if result:
                st.subheader("📝 Transcription Results")
                
                if processing_method == "Detailed Progress (Chunk-based)":
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Conversion Time", f"{conversion_time:.2f}s")
                    with col2:
                        st.metric("Total Chunks", result.get('total_chunks', 'N/A'))
                    with col3:
                        st.metric("Successful", result.get('successful_chunks', 'N/A'))
                    with col4:
                        st.metric("Failed", result.get('failed_chunks', 'N/A'))
                    
                    transcription_text = result['text']
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Conversion Time", f"{conversion_time:.2f} seconds")
                    with col2:
                        st.metric("Words Count", len(result['text'].split()))
                    
                    transcription_text = result['text']
                
                st.text_area("Transcribed Text:", transcription_text, height=150)
                
                if processing_method == "Detailed Progress (Chunk-based)" and 'chunks' in result:
                    with st.expander("🕐 View Chunk Details"):
                        for chunk in result['chunks']:
                            if chunk['success']:
                                st.success(f"[{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s]: {chunk['text']}")
                            else:
                                st.error(f"[{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s]: {chunk['text']}")

            try:
                os.unlink(tmp_file_path)
            except:
                pass

# Option 2: Upload audio
elif audio_option == 'Upload Audio':
    audio_file = st.file_uploader("📁 Upload an audio file", type=["wav", "mp3", "ogg", "m4a", "flac"])

    if audio_file:
        st.success(f"✅ File uploaded: {audio_file.name}")
        
        with st.spinner("🔄 Loading audio file..."):
            audio_data, sr = librosa.load(audio_file, sr=None)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{audio_file.size:,} bytes")
        with col2:
            st.metric("Sample Rate", f"{sr:,} Hz")
        with col3:
            st.metric("Duration", f"{len(audio_data) / sr:.2f} sec")

        st.audio(audio_file, format=f'audio/{audio_file.type.split("/")[-1]}')

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sr, format='WAV')
            tmp_file_path = tmp_file.name
            
        if st.button("🚀 Start Conversion", type="primary"):
            st.subheader("🔄 Audio Conversion Status")
            
            if processing_method == "Detailed Progress (Chunk-based)":
                result, conversion_time = convert_audio_to_text_with_chunks(tmp_file_path, model_choice, sr, is_file_path=True)
            else:
                result, conversion_time = convert_audio_to_text_simple(tmp_file_path, model_choice, is_file_path=True)

            if result:
                st.subheader("📝 Transcription Results")
                
                if processing_method == "Detailed Progress (Chunk-based)":
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Conversion Time", f"{conversion_time:.2f}s")
                    with col2:
                        st.metric("Total Chunks", result.get('total_chunks', 'N/A'))
                    with col3:
                        st.metric("Successful", result.get('successful_chunks', 'N/A'))
                    with col4:
                        st.metric("Failed", result.get('failed_chunks', 'N/A'))
                    
                    transcription_text = result['text']
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Conversion Time", f"{conversion_time:.2f} seconds")
                    with col2:
                        st.metric("Words Count", len(result['text'].split()))
                    
                    transcription_text = result['text']
                
                st.text_area("Transcribed Text:", transcription_text, height=150)
                
                st.download_button(
                    label="📥 Download Transcription",
                    data=transcription_text,
                    file_name=f"transcription_{audio_file.name}.txt",
                    mime="text/plain"
                )
                
                if processing_method == "Detailed Progress (Chunk-based)" and 'chunks' in result:
                    with st.expander("🕐 View Chunk Details"):
                        for chunk in result['chunks']:
                            if chunk['success']:
                                st.success(f"[{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s]: {chunk['text']}")
                            else:
                                st.error(f"[{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s]: {chunk['text']}")

        try:
            os.unlink(tmp_file_path)
        except:
            pass

else:
    st.write("Please select an audio input option.")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Information")
    st.write("""
    **Processing Methods:**
    - **Detailed Progress**: Shows chunk-by-chunk progress with live transcription results
    - **Simple Progress**: Faster processing with basic progress indication
    
    **Supported formats:**
    - WAV, MP3, OGG, M4A, FLAC
    
    **Tips:**
    - Use Detailed Progress for long audio files to track conversion progress
    - Simple Progress is faster for shorter files
    - Clear audio gives better results
    """)
    
    st.header("🎯 Current Selection")
    st.write(f"**Language:** {language}")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Method:** {processing_method}")