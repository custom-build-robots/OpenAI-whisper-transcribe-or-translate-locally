# OpenAI-whisper-transcribe-or-translate-locally

This Python script uses OpenAI's Whisper model to transcribe or translate audio files. It allows you to optionally set the source language, specify a translation task, and customize the output file name. By default, the script assumes English as the source language and transcribes the audio.

On my blog you will get some more details about the program and how to use it:
[How to Install and Use OpenAI’s Whisper Locally for Automatic Transcription and Translation](https://ai-box.eu/en/news/how-to-install-and-use-openais-whisper-locally-for-automatic-transcription-and-translation/1449/)

## Features

- **Audio Transcription**: Converts audio files into text.
- **Audio Translation**: Translates audio content into English (optional).
- **Custom Source Language**: Supports a wide range of languages.
- **Customizable Output**: Specify a custom output file name or use the default naming convention.
- **Model Caching**: Downloads and caches Whisper models in a local directory for offline use.


## Whisper Installation

Install Whisper and its dependencies locally: Use the installation script **install_whisper.sh** and HowTo provided in my repository:

[Installation Scripts for Generative AI Tools](https://github.com/custom-build-robots/Installation-Scripts-for-Generative-AI-Tools)


## Usage whisper_transcribe_translate.py terminal
1. Now donwload the whisper_transcribe_translate.py to run it locally on your system in the active virtual whisper environment:
   ```bash
   wget https://raw.githubusercontent.com/custom-build-robots/OpenAI-whisper-transcribe-or-translate-locally/refs/heads/main/whisper_transcribe_translate.py
2. Basic Usage:

   Run the script to transcribe an English audio file like a locally stored mp3 file:
   ```bash
   python whisper_transcribe_translate.py

## Usage whisper_transcribe_translate_processing.py gradio UI
1. Now donwload the whisper_transcribe_translate_processing.py to run it locally on your system in the active virtual whisper environment:
   ```bash
   wget https://raw.githubusercontent.com/custom-build-robots/OpenAI-whisper-transcribe-or-translate-locally/refs/heads/main/whisper_transcribe_translate_processing.py
2. Basic Usage:

   Run the script to transcribe an English audio file like a locally stored mp3 files or video files like mp4:
   ```bash
   python whisper_transcribe_translate_processing.py


## OpenAI Whisper: Transcribe & Translate Web App

![OpenAI's Whisper Transcribe & Translate Web App](https://ai-box.eu/wp-content/uploads/2024/12/OpenAIs_Whisper_transcribe_translate_web_app-978x1024.png)

This project provides a web-based interface to OpenAI's Whisper model, allowing users to transcribe or translate audio files locally with a user-friendly interface and to download videos and process them afterwards.

   
## Whisper Model Management
The script uses the Whisper large-v2 model by default, but you can customize the model or specify its directory.
I already successfully used the following models locally on my server:
- base.pt  
- large-v2.pt
- large-v3.pt

Models are cached in a local models/ folder.

## Contributing
Feel free to open issues or submit pull requests to improve the script.
