#!/usr/bin/env python3
# Author: Ingmar Stapel
# Date: 2024-12-29
# Version: 0.4
#
# Description:
# This script provides a Gradio-based user interface to transcribe or translate audio (and now also
# video) files locally using OpenAI's Whisper model. It has two main features:
#   1) Processing a local file (audio or video).
#   2) Processing a video URL (e.g., YouTube link), downloading the video, extracting its audio,
#      and then running Whisper on the extracted audio.
#
# Users can specify:
#   - The Whisper model name (e.g., 'base', 'large', or 'large-v2').
#   - The source language (defaults to English).
#   - The task: either 'transcribe' or 'transcribe & translate' (the latter maps to Whisper's 'translate' task).
#
# All Whisper models are downloaded/cached locally to the 'models/' directory. The script also
# stores transcriptions and translations in a timestamped .txt file inside the 'transcribe/' folder.
#
# Requirements:
#   - Python 3.7+
#   - The 'whisper' package (OpenAI’s Whisper)
#   - 'gradio' for the user interface
#   - 'yt-dlp' for video downloading (for the "Process Video URL" feature)
#   - 'ffmpeg' for audio extraction
#   - Other dependencies listed in the requirements file or installation instructions
#
# Run the script locally with:
#   python3 <script_name>.py
#
# Once launched, open the provided local URL in your web browser to use the interface.
#
# Note: By default, the script assumes English for the source language and performs transcription.
#       You may specify a different language or task type in the Gradio UI.

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

def process_audio(input_file, model_name, source_language, task):
    """
    Processes a user-uploaded file. If it's audio, proceed directly;
    if it's a video, extract the audio with ffmpeg first, then transcribe.
    """

    model_path = "models/"
    output_folder = "transcribe/"

    # Ensure our directories exist
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Detect file extension
    file_ext = os.path.splitext(input_file)[1].lower()

    # List of "known" audio formats. If extension not in this list, we'll assume it's video.
    known_audio_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]

    # If this is a "video," extract the audio first
    if file_ext not in known_audio_exts:
        # Generate a timestamped WAV name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_audio = os.path.join(output_folder, f"extracted_{timestamp}.wav")

        # Extract audio with ffmpeg
        # -vn means "no video", output WAV at 16-bit, 44.1 kHz (modify if you prefer)
        extract_cmd = [
            "ffmpeg", "-i", input_file, "-vn",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            extracted_audio
        ]
        try:
            subprocess.run(extract_cmd, check=True)
            # Use the extracted WAV as the "audio_file" for Whisper
            audio_file = extracted_audio
        except subprocess.CalledProcessError as e:
            return (
                f"Failed to extract audio: {str(e)}",
                "",
                "Audio extraction failed.",
                ""
            )
    else:
        # It's already audio
        audio_file = input_file

    # Load the Whisper model
    model = whisper.load_model(model_name, download_root=model_path)
    model_file = os.path.join(model_path, f"{model_name}.pt")

    if os.path.exists(model_file):
        model_status = f"Model successfully loaded from: {model_file}"
    else:
        model_status = f"Model was downloaded to: {model_file}"

    # If task is "transcribe & translate," map to Whisper's internal "translate"
    if task == "transcribe & translate":
        task = "translate"

    # Now run Whisper on the (possibly extracted) audio
    result = model.transcribe(audio_file, language=source_language, task=task)
    output_text = result["text"]

    # Compose an output text filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_type = "translated" if task == "translate" else "transcribed"

    output_filename = os.path.join(
        output_folder,
        f"{base_name}_{output_type}_{timestamp}.txt"
    )

    # Save the transcription
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(output_text)

    return output_text, model_status, f"Output saved to {output_filename}", output_filename


def process_url(video_url, model_name, source_language, task):
    """
    Downloads a video from a given URL, appending a timestamp to the filename,
    extracts its audio, and runs Whisper.
    """
    model_path = "models/"
    output_folder = "transcribe/"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Generate a timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Download video with yt-dlp
        download_command = [
            "yt-dlp",
            "-o",
            os.path.join(output_folder, f"%(title)s_%(id)s_{timestamp}.%(ext)s"),
            video_url
        ]
        subprocess.run(download_command, check=True)

        # Identify the downloaded *.mp4 file
        video_filename = None
        video_id = None
        for file in os.listdir(output_folder):
            if file.endswith(".mp4") and timestamp in file:
                video_filename = os.path.join(output_folder, file)
                # Attempt to parse out a video ID if desired
                parts = os.path.splitext(file)[0].split('_')
                video_id = parts[-2] if len(parts) >= 2 else None
                break

        if not video_filename:
            return None, None, "No video file found after download.", None

        # Extract audio with ffmpeg, add timestamp to the mp3
        audio_filename = os.path.splitext(video_filename)[0] + f"_{timestamp}.mp3"
        subprocess.run(["ffmpeg", "-i", video_filename, "-q:a", "0", "-map", "a", audio_filename], check=True)

    except subprocess.CalledProcessError as e:
        return None, None, f"Error during video or audio processing: {str(e)}", None

    # Load the Whisper model
    model = whisper.load_model(model_name, download_root=model_path)
    model_file = os.path.join(model_path, f"{model_name}.pt")
    if os.path.exists(model_file):
        model_status = f"Model successfully loaded from: {model_file}"
    else:
        model_status = f"Model was downloaded to: {model_file}"

    if task == "transcribe & translate":
        task = "translate"

    # Transcribe the extracted audio
    result = model.transcribe(audio_filename, language=source_language, task=task)
    output_text = result["text"]

    # Create a text file for the final transcription
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    output_type = "translated" if task == "translate" else "transcribed"

    if video_id:
        output_filename = os.path.join(
            output_folder,
            f"{base_name}_{output_type}_{final_timestamp}_{video_id}.txt"
        )
    else:
        output_filename = os.path.join(
            output_folder,
            f"{base_name}_{output_type}_{final_timestamp}.txt"
        )

    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(output_text)

    return output_text, model_status, f"Output saved to {output_filename}", output_filename

def gradio_app():
    try:
        with gr.Blocks() as app:
            gr.Markdown("""
        # Whisper Transcribe & Translate
        Transcribe or translate audio **or video** files locally using OpenAI's Whisper model. 
        
        **Tab: Process Audio or Video File**
          - You can upload an audio file (e.g., WAV, MP3) or a video file (e.g., MP4). If a video is detected, 
            this app will automatically extract its audio via ffmpeg before transcribing or translating.

        **Tab: Process Video URL**
          - Easily download a video from supported websites and let the app handle the transcription or translation for you.
          - Check out the list of [Supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) 
            for video downloading. 
        """)

            with gr.Tab("Process Audio or Video File"):
                gr.Markdown("## Upload an audio or video file")
                with gr.Row():
                    uploaded_file = gr.File(label="Upload Audio/Video File")

                with gr.Row():
                    model_name_audio = gr.Dropdown(
                        choices=["base", "large", "large-v2"],
                        value="base",
                        label="Whisper Model Name"
                    )
                    source_language_audio = gr.Textbox(
                        label="Source Language (leave empty for English)",
                        value="English"
                    )
                    task_audio = gr.Radio(
                        choices=["transcribe", "transcribe & translate"],
                        value="transcribe",
                        label="Task"
                    )

                submit_audio = gr.Button("Process File")

                output_text_audio = gr.Textbox(label="Transcription/Translation Output")
                model_status_audio = gr.Textbox(label="Model Status")
                file_status_audio = gr.Textbox(label="File Save Status")
                download_button_audio = gr.File(label="Download Output File")

                submit_audio.click(
                    fn=process_audio,
                    inputs=[uploaded_file, model_name_audio, source_language_audio, task_audio],
                    outputs=[output_text_audio, model_status_audio, file_status_audio, download_button_audio]
                )

            with gr.Tab("Process Video URL"):
                gr.Markdown("## Paste a Video URL to process its audio")
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
                    fn=process_url,
                    inputs=[video_url, model_name, source_language, task],
                    outputs=[output_text, model_status, file_status, download_button]
                )

            app.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Starting Whisper Gradio app...")
    gradio_app()
