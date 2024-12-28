# OpenAI-whisper-transcribe-or-translate-locally

This Python script uses OpenAI's Whisper model to transcribe or translate audio files. It allows you to optionally set the source language, specify a translation task, and customize the output file name. By default, the script assumes English as the source language and transcribes the audio.

## Features

- **Audio Transcription**: Converts audio files into text.
- **Audio Translation**: Translates audio content into English (optional).
- **Custom Source Language**: Supports a wide range of languages.
- **Customizable Output**: Specify a custom output file name or use the default naming convention.
- **Model Caching**: Downloads and caches Whisper models in a local directory for offline use.


## Installation

Install Whisper and its dependencies locally: Use the installation script **install_whisper.sh** provided in my repository:

[Installation Scripts for Generative AI Tools](https://github.com/custom-build-robots/Installation-Scripts-for-Generative-AI-Tools)


### Download the script as follows:

To install whisper locally on your Ubuntu machine follow the link below in the instruction provided

   ```bash
   wget https://raw.githubusercontent.com/custom-build-robots/Installation-Scripts-for-Generative-AI-Tools/main/install_whisper.sh


After installing whisper start the virtual environment which was created and follow the description below.

## Usage
Now donwload the whisper_transcribe_translate.py to run it locally on your system in the active virtual whisper environment.

   ```bash
   wget <add the file>


Basic Usage: Run the script to transcribe an English audio file:

   ```bash
   python whisper_transcribe_translate.py


## Whisper Model Management
The script uses the Whisper large-v2 model by default, but you can customize the model or specify its directory. Models are cached in a local models/ folder.

## Contributing
Feel free to open issues or submit pull requests to improve the script.
