# OpenAI-whisper-transcribe-or-translate-locally

This Python script uses OpenAI's Whisper model to transcribe or translate audio files. It allows you to optionally set the source language, specify a translation task, and customize the output file name. By default, the script assumes English as the source language and transcribes the audio.

## Features

- **Audio Transcription**: Converts audio files into text.
- **Audio Translation**: Translates audio content into English (optional).
- **Custom Source Language**: Supports a wide range of languages.
- **Customizable Output**: Specify a custom output file name or use the default naming convention.
- **Model Caching**: Downloads and caches Whisper models in a local directory for offline use.

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - `whisper`
  - `torch`
  - (Optional for translation into non-English languages): `googletrans`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/whisper-transcribe-translate.git
   cd whisper-transcribe-translate
