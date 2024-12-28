#!/usr/bin/env python3
# Author: Ingmar Stapel
# Date: 2024-12-28
# Version: 0.4
# The script is used to transcribe or translate audio files locally using OpenAI's Whisper model.
# It allows the user to specify the source language, task (transcription or translation), 
# and customize the output file name. By default, the script assumes English as the source language 
# and performs transcription. 
# It also supports downloading and caching Whisper models locally.

import os
import whisper
import torch
import warnings

# Suppress FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Dynamically patch `torch.load` to include weights_only=True
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Set weights_only=True if not explicitly provided
    kwargs.setdefault("weights_only", True)
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Define the custom directory for models
model_path = "models/"

# Use "base", "large", or "large-v2" depending on your need
model_name = "base"

print(f"The default file name for the output is: {model_name}")
model_name = input("Enter your Whisper model name (base, large, large-v2...): ") or model_name

# Ensure the custom model folder exists
os.makedirs(model_path, exist_ok=True)

# Load the Whisper model, specifying the custom download directory
print(f"Loading model '{model_name}' from directory: {model_path}")
model = whisper.load_model(model_name, download_root=model_path)

# Verify if the model was used from the custom folder
model_file = os.path.join(model_path, f"{model_name}.pt")
if os.path.exists(model_file):
    print(f"Model successfully loaded from: {model_file}")
else:
    print(f"Model was downloaded to: {model_file}")

# Specify the path to your audio file
audio_file = input("Enter source audio file (/home/ingmar/whisper_offline/Satya.mp3): ")

# Optional: Set the source language and task
# Defaults: English as the source language and transcription as the task
source_language = input("Enter source language (leave empty for English): ") or "English"
task = input("Enter task ('transcribe' or 'translate', leave empty for 'transcribe'): ") or "transcribe"

print(f"Processing audio file: {audio_file}")
print(f"Source language: {source_language}")
print(f"Task: {task}")

# Transcribe or translate the audio based on user input
result = model.transcribe(audio_file, language=source_language, task=task)

# Get the transcription/translation text
output_text = result["text"]

# Get the base name of the audio file (without extension)
base_name = os.path.splitext(audio_file)[0]

# Define the default file name
output_type = "translated" if task == "translate" else "transcribed"
default_text_file = f"{base_name}_{output_type}.txt"

# Prompt the user for a custom file name or use the default
print(f"The default file name for the output is: {default_text_file}")
custom_text_file = input("Enter a custom file name (leave empty to use the default): ") or default_text_file

# Save the transcription/translation to the specified text file
with open(custom_text_file, "w", encoding="utf-8") as file:
    file.write(output_text)

print(f"Output saved to {custom_text_file}")
