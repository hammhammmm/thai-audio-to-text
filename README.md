# 🎙️ Audio to Text Converter

A powerful Streamlit application that converts audio files and live recordings to text using state-of-the-art speech recognition models from Hugging Face Transformers.

## ✨ Features

- **Multiple ASR Models**: Support for Whisper, Wav2Vec2, and other popular models
- **Real-time Recording**: Record audio directly from your browser
- **File Upload**: Support for various audio formats (WAV, MP3, OGG, M4A, FLAC)
- **Detailed Progress Tracking**: Chunk-by-chunk conversion with live progress updates
- **Multi-language Support**: English, Thai, and multilingual models
- **Live Transcription**: See results as they're processed
- **Download Results**: Export transcriptions as text files
- **Timestamp Display**: View time-aligned transcription chunks

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- A modern web browser with microphone access (for recording)

### Step 1: Clone or Download the Code

```bash
# Option 1: If you have git installed
git clone https://github.com/hammhammmm/thai-audio-to-text
cd audio-to-text-converter

# Option 2: Download the Python file directly
# Save the code as 'app.py' in a new folder
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv audio_converter_env

# Activate virtual environment
# On Windows:
audio_converter_env\Scripts\activate

# On macOS/Linux:
source audio_converter_env/bin/activate
```

### Step 3: Install Required Packages

```bash
# Install all required dependencies
pip install streamlit
pip install soundfile
pip install librosa
pip install numpy
pip install transformers
pip install streamlit-webrtc
pip install av
pip install torch torchvision torchaudio
```

**Alternative: Install from requirements file**

Create a `requirements.txt` file with the following content:

```text
streamlit>=1.28.0
soundfile>=0.12.1
librosa>=0.10.1
numpy>=1.24.0
transformers>=4.35.0
streamlit-webrtc>=0.47.0
av>=10.0.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Test if all packages are installed correctly:

```bash
python -c "import streamlit, soundfile, librosa, transformers, streamlit_webrtc; print('All packages installed successfully!')"
```

## 🚀 Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### First Time Setup

When you first run the app, it will download the required ML models from Hugging Face. This may take a few minutes depending on your internet connection.

## 📖 How to Use

### 1. **Select Your Preferences**

- **Choose Language**: Select "English" or "Thai"
- **Choose Model**: Pick from available ASR models:
  - Whisper (English) - Fast, English-only
  - Whisper (Multilingual) - Supports multiple languages
  - Whisper Large V3 - Highest accuracy
  - Facebook Wav2Vec2 - Good for English
  - Google Wav2Vec2 - Multilingual support

### 2. **Choose Processing Method**

- **Detailed Progress (Chunk-based)**: 
  - Shows second-by-second conversion progress
  - Best for long audio files
  - Provides chunk-by-chunk results
  - Takes longer but more informative

- **Simple Progress**:
  - Faster processing
  - Basic progress indication
  - Best for short audio files

### 3. **Input Your Audio**

#### Option A: Record Audio
1. Click "Record Audio" option
2. Click the record button to start recording
3. Speak clearly into your microphone
4. Click stop when finished
5. The app will automatically process your recording

#### Option B: Upload Audio File
1. Click "Upload Audio" option
2. Click "Browse files" and select your audio file
3. Supported formats: WAV, MP3, OGG, M4A, FLAC
4. Click "🚀 Start Conversion" to begin processing

### 4. **View Results**

- **Live Progress**: Watch real-time conversion progress
- **Transcription**: Read the converted text
- **Statistics**: View processing time and accuracy metrics
- **Download**: Save transcription as a text file
- **Timestamps**: Expand to see time-aligned text chunks

## 🎯 Model Recommendations

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| English podcasts/interviews | Whisper Large V3 | Highest accuracy for English |
| Quick English transcription | Whisper (English) | Fast and efficient |
| Multi-language content | Whisper (Multilingual) | Supports 99+ languages |
| Thai language | Whisper (Thai) | Optimized for Thai speech |
| Academic/Research | Facebook Wav2Vec2 | Good balance of speed and accuracy |

## 📁 File Format Support

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Best quality, recommended |
| MP3 | `.mp3` | Most common format |
| OGG | `.ogg` | Open source format |
| M4A | `.m4a` | Apple format |
| FLAC | `.flac` | Lossless compression |

## ⚡ Performance Tips

### For Best Results:
- **Clear Audio**: Use high-quality recordings without background noise
- **Good Microphone**: Use a decent microphone for recording
- **Optimal Length**: 1-30 minutes work best (longer files take more time)
- **Stable Internet**: Required for downloading models

### For Faster Processing:
- Use "Simple Progress" method for short files
- Choose smaller models (Whisper English vs Whisper Large V3)
- Ensure sufficient RAM (8GB+ recommended)

## 🔧 Troubleshooting

### Common Issues and Solutions:

**1. "ModuleNotFoundError" when running the app**
```bash
# Make sure virtual environment is activated
source audio_converter_env/bin/activate  # macOS/Linux
# or
audio_converter_env\Scripts\activate     # Windows

# Reinstall missing package
pip install [missing-package-name]
```

**2. "Recording not working"**
- Check browser permissions for microphone access
- Try using Chrome or Firefox browsers
- Ensure microphone is not being used by other applications

**3. "Model download fails"**
- Check internet connection
- Try switching to a different model
- Clear browser cache and restart the app

**4. "Out of memory" errors**
- Close other applications
- Try using a smaller model
- Use "Simple Progress" method
- Process shorter audio files

**5. "Conversion is very slow"**
- Use GPU acceleration if available:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Choose faster models like "Whisper (English)"
- Use "Simple Progress" method

### System Requirements:

**Minimum:**
- 4GB RAM
- 2GB free disk space
- Python 3.8+

**Recommended:**
- 8GB+ RAM
- 5GB+ free disk space
- GPU with CUDA support (optional, for faster processing)
- Python 3.9+

## 🔒 Privacy & Data

- **No data is stored**: Audio files are processed locally and deleted after conversion
- **No cloud processing**: All speech recognition happens on your machine
- **Temporary files**: Only created during processing and automatically cleaned up
- **Model downloads**: Models are downloaded from Hugging Face and cached locally

## 🆘 Getting Help

If you encounter issues:

1. **Check the terminal/console** for error messages
2. **Restart the application**: Stop with Ctrl+C and run again
3. **Clear browser cache** and refresh the page
4. **Try a different browser** (Chrome, Firefox recommended)
5. **Check system resources** (RAM, disk space)

## 🔄 Updating

To update the application:

```bash
# Activate your virtual environment
source audio_converter_env/bin/activate

# Update packages
pip install --upgrade streamlit transformers librosa soundfile

# Restart the application
streamlit run app.py
```

## 📝 Example Usage

```bash
# 1. Activate environment
source audio_converter_env/bin/activate

# 2. Run the app
streamlit run app.py

# 3. Open browser to http://localhost:8501

# 4. Select "English" language and "Whisper Large V3" model

# 5. Upload an audio file or record directly

# 6. Choose "Detailed Progress" to see chunk-by-chunk conversion

# 7. Download your transcription when complete
```

## 🎉 You're Ready!

Your Audio to Text Converter is now ready to use. Start by uploading a short audio file to test the setup, then experiment with different models and processing methods to find what works best for your needs.

---

**Happy Transcribing! 🎤➡️📝**