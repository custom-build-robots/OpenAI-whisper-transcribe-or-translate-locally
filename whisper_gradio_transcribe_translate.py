import os
import whisper
import torch
import warnings
import gradio as gr
from datetime import datetime

# Suppress FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Dynamically patch `torch.load` to include weights_only=True
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Set weights_only=True if not explicitly provided
    kwargs.setdefault("weights_only", True)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Function to transcribe or translate audio
def process_audio(audio_file, model_name, source_language, task):
    model_path = "models/"
    output_folder = "transcribe/"

    # Ensure the custom model and output folders exist
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

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
    result = model.transcribe(audio_file, language=source_language, task=task)
    output_text = result["text"]

    # Set the output file name with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_type = "translated" if task == "translate" else "transcribed"
    output_filename = os.path.join(output_folder, f"{base_name}_{output_type}_{timestamp}.txt")

    # Save the transcription/translation to a text file
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(output_text)

    return output_text, model_status, f"Output saved to {output_filename}", output_filename

# Gradio interface
def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("""
        # OpenAI's Whisper Transcribe & Translate Web-App (all local)
        Transcribe or translate audio files locally using OpenAI's Whisper model. Configure your settings below.
        """)

        with gr.Row():
            audio_file = gr.File(label="Audio File", type="filepath")

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

        submit = gr.Button("Process Audio")

        output_text = gr.Textbox(label="Transcription/Translation Output")
        model_status = gr.Textbox(label="Model Status")
        file_status = gr.Textbox(label="File Save Status")
        download_button = gr.File(label="Download Output File")

        submit.click(
            process_audio,
            inputs=[audio_file, model_name, source_language, task],
            outputs=[output_text, model_status, file_status, download_button]
        )

    return app

# Run the Gradio app
if __name__ == "__main__":
    app = gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)

