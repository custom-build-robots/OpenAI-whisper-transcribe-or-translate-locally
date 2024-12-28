import os
import whisper
import torch
import warnings
import gradio as gr
from datetime import datetime
import subprocess

# Suppress FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Dynamically patch `torch.load` to include weights_only=True
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Set weights_only=True if not explicitly provided
    kwargs.setdefault("weights_only", True)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Function to download video, extract audio, and process

def process_url(video_url, model_name, source_language, task):
    model_path = "models/"
    output_folder = "transcribe/"

    # Ensure the custom model and output folders exist
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Download the video using yt-dlp
        download_command = [
            "yt-dlp",
            "-o",
            os.path.join(output_folder, "%(title)s_%(id)s.%(ext)s"),
            video_url
        ]
        subprocess.run(download_command, check=True)

        # Identify the downloaded video file
        video_filename = None
        video_id = None
        for file in os.listdir(output_folder):
            if file.endswith(".mp4"):
                video_filename = os.path.join(output_folder, file)
                video_id = os.path.splitext(file)[0].split('_')[-1]  # Extract YouTube ID
                break
        if not video_filename:
            return None, None, "No video file found after download.", None

        # Extract audio using ffmpeg
        audio_filename = os.path.splitext(video_filename)[0] + ".mp3"
        subprocess.run(["ffmpeg", "-i", video_filename, "-q:a", "0", "-map", "a", audio_filename], check=True)
    except subprocess.CalledProcessError as e:
        return None, None, f"Error during video or audio processing: {str(e)}", None

    # Load the Whisper model
    model = whisper.load_model(model_name, download_root=model_path)

    # Verify if the model was used from the custom folder
    model_file = os.path.join(model_path, f"{model_name}.pt")
    if os.path.exists(model_file):
        model_status = f"Model successfully loaded from: {model_file}"
    else:
        model_status = f"Model was downloaded to: {model_file}"

    # Map "transcribe & translate" to Whisper's "translate" task
    if task == "transcribe & translate":
        task = "translate"

    # Process the audio file
    result = model.transcribe(audio_filename, language=source_language, task=task)
    output_text = result["text"]

    # Set the output file name with a timestamp and video ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    output_type = "translated" if task == "translate" else "transcribed"
    output_filename = os.path.join(output_folder, f"{base_name}_{output_type}_{timestamp}_{video_id}.txt")

    # Save the transcription/translation to a text file
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(output_text)

    return output_text, model_status, f"Output saved to {output_filename}", output_filename

# Gradio interface
def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("""
        # Whisper Transcribe & Translate
        Transcribe or translate audio files locally using OpenAI's Whisper model. Configure your settings below.
        """)

        with gr.Row():
            video_url = gr.Textbox(label="Video URL")

        with gr.Row():
            model_name = gr.Dropdown(
                choices=["base", "large", "large-v2"],
                value="base",
                label="Whisper Model Name"
            )
            source_language = gr.Textbox(
                label="Source Language (leave empty for English)",
                value="English"
            )
            task = gr.Radio(
                choices=["transcribe", "transcribe & translate"],
                value="transcribe",
                label="Task"
            )

        submit = gr.Button("Process Video URL")

        output_text = gr.Textbox(label="Transcription/Translation Output")
        model_status = gr.Textbox(label="Model Status")
        file_status = gr.Textbox(label="File Save Status")
        download_button = gr.File(label="Download Output File")

        submit.click(
            process_url,
            inputs=[video_url, model_name, source_language, task],
            outputs=[output_text, model_status, file_status, download_button]
        )

    return app

# Run the Gradio app
if __name__ == "__main__":
    app = gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)

