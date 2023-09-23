import gradio as gr
import requests
import os

# Define the Whisper ASR function (transcribe_audio) here

# Retrieve the API token from the environment variable
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Check if the API token is available
if not API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set.")

# Define the BART summarization API URL
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def summarize_video(youtube_url: str, task: str = "transcribe", return_timestamps: bool = False, summary_length: int = 150) -> dict:
    # Call your transcribe_audio function to get the transcription
    transcription_result = transcribe_audio(youtube_url, task, return_timestamps)
    
    # Summarize the transcription
    summary_result = query({
        "inputs": transcription_result["transcription"][:summary_length]
    })
    
    return summary_result

MODEL_NAME = "openai/whisper-large-v2"

EXAMPLES = [
    ["https://www.youtube.com/watch?v=H1YoNlz2LxA", "translate", False],
]

# Define the Gradio interface for transcription
yt_transcribe = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
        gr.inputs.Checkbox(label="Return timestamps"),
    ],
    outputs=[gr.outputs.HTML(label="Video"),
        gr.outputs.Textbox(label="Transcription").style(show_copy_button=True)],
    layout="horizontal",
    theme=gr.themes.Base(),
    title="Whisper Large V2: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the checkpoint"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe video files of"
        " arbitrary length."
    ),
    allow_flagging="never",
    examples=EXAMPLES,
    cache_examples=False
)

# Define the Gradio interface for summarization
yt_summarize = gr.Interface(
    fn=summarize_video,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
        gr.inputs.Checkbox(label="Return timestamps"),
        gr.inputs.Number(default=150, label="Summary Length", min=1, max=500),
    ],
    outputs=[gr.outputs.HTML(label="Video"),
        gr.outputs.Textbox(label="Summary").style(show_copy_button=True)],
    layout="horizontal",
    theme=gr.themes.Base(),
    title="Whisper Large V2: Summarize YouTube",
    description=(
        "Summarize long-form YouTube videos with the click of a button! This tab uses the Whisper ASR model for transcription"
        " and BART for summarization."
    ),
    allow_flagging="never",
    examples=EXAMPLES,
    cache_examples=False
)

# Add the "Summarize" tab to the Gradio interface
yt_transcribe.tabs["Summarize"] = yt_summarize

# Launch the Gradio interface
with yt_transcribe:
    gr.DuplicateButton()
    gr.TabbedInterface([yt_transcribe], ["YouTube"])
    yt_transcribe.launch(enable_queue=True)
