# OpenAI-whisper-transcribe-or-translate-locally

This Python script uses OpenAI's Whisper model to transcribe or translate audio files. It allows you to optionally set the source language, specify a translation task, and customize the output file name. By default, the script assumes English as the source language and transcribes the audio.

## Features

- **Audio Transcription**: Converts audio files into text.
- **Audio Translation**: Translates audio content into English (optional).
- **Custom Source Language**: Supports a wide range of languages.
- **Customizable Output**: Specify a custom output file name or use the default naming convention.
- **Model Caching**: Downloads and caches Whisper models in a local directory for offline use.


## Installation

Install Whisper and its dependencies locally: Use the installation script provided in my repository:

Link: 

### Download the script as follows:

wget https://raw.githubusercontent.com/custom-build-robots/Installation-Scripts-for-Generative-AI-Tools/main/install_whisper.sh

## Usage
Basic Usage: Run the script to transcribe an English audio file:
python whisper_transcribe_translate.py

## Whisper Model Management
The script uses the Whisper large-v2 model by default, but you can customize the model or specify its directory. Models are cached in a local models/ folder.

## Contributing
Feel free to open issues or submit pull requests to improve the script.
