import os
import whisper

# Define the custom directory for models
model_path = "models/"

# Use "base", "large", or "large-v2" depending on your need
model_name = "base"

# Ensure the custom model folder exists
os.makedirs(model_path, exist_ok=True)

# Load the Whisper model, specifying the custom download directory
print(f"Loading model '{model_name}' from directory: {model_path}")
model = whisper.load_model(model_name, download_root=model_path)

# Specify the path to your audio file
audio_file = "/home/ingmar/whisper_offline/Satya.mp3"  # Replace with your MP3 file path

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

# Verify if the model was used from the custom folder
model_file = os.path.join(model_path, f"{model_name}.pt")
if os.path.exists(model_file):
    print(f"Model successfully loaded from: {model_file}")
else:
    print(f"Model was downloaded to: {model_file}")

