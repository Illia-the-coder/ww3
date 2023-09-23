
import gradio as gr

import os
from gradio_client import Client

def transcribe_audio(youtube_url: str, task: str = "transcribe", return_timestamps: bool = True, api_name: str = "/predict_2") -> dict:
    """
    Transcribe audio from a given YouTube URL using a specified model.

    Parameters:
    - youtube_url (str): The YouTube URL to transcribe.
    - task (str, optional): The task to perform. Default is "transcribe".
    - return_timestamps (bool, optional): Whether to return timestamps. Default is True.
    - api_name (str, optional): The API endpoint to use. Default is "/predict_2".

    Returns:
    - dict: The transcription result.
    """
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    result = client.predict(youtube_url, task, return_timestamps, api_name)
    return result



MODEL_NAME = "openai/whisper-large-v2"


demo = gr.Blocks()



yt_transcribe = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe")
    ],
    outputs=["html", "text"],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V2: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the checkpoint"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe video files of"
        " arbitrary length."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([yt_transcribe], [ "YouTube"])

demo.launch(enable_queue=True)

