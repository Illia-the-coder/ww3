import gradio as gr
import os
import requests
from gradio_client import Client

# Define the Hugging Face API URL and headers with your token
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Replace with your actual token
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def transcribe_and_summarize(youtube_url: str, task: str = "transcribe", return_timestamps: bool = False, summarize: bool = False, api_name: str = "/predict_2") -> dict:
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    result = list(client.predict(youtube_url, task, return_timestamps, fn_index=7))
    
    # If the "summarize" checkbox is selected, summarize the transcription
    if summarize:
        transcription = result[1]
        summary_result = query({"inputs": transcription})
        try:
            result[2] = summary_result[0]['summary_text']
        except:
            result[2] = 'Model is overloaded.'
            
    else:
        result[2] = ''
        
    return result

MODEL_NAME = "openai/whisper-large-v2"

demo = gr.Blocks()

EXAMPLES = [
    ["https://www.youtube.com/watch?v=HyBw3wcZ124", "transcribe", False],
]

# Define the Gradio interface with the "Summarize" checkbox and "Summary" output
yt_transcribe = gr.Interface(
    fn=transcribe_and_summarize,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
        gr.inputs.Checkbox(label="Return timestamps"),
        gr.inputs.Checkbox(label="Summarize")  # Added "Summarize" checkbox
    ],
    outputs=[
        gr.outputs.HTML(label="Video"),
        gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
        gr.outputs.Textbox(label="Summary").style(show_copy_button=True)  # Added "Summary" output
    ],
    layout="horizontal",
    theme=gr.themes.Base(),
    title="Whisper Large V2: Transcribe YouTube with Summarization",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the checkpoint"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe video files of"
        " arbitrary length."
    ),
    allow_flagging="never",
    examples=EXAMPLES,
    cache_examples=False
)

with demo:
    gr.DuplicateButton()
    gr.TabbedInterface([yt_transcribe], ["YouTube"])

demo.launch(enable_queue=True)
