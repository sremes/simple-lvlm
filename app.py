from model import LVLM

import gradio as gr
import torch
from pathlib import Path
from PIL import Image

model: LVLM = None


def load_model(checkpoint_path: Path) -> LVLM:
    global model
    model = LVLM(
        language_model="stabilityai/stablelm-3b-4e1t",
        vision_model="openai/clip-vit-large-patch14",
    )
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"], strict=False)


def load_image(image_path: Path) -> torch.Tensor:
    global model
    image = Image.open(image_path)
    return model.image_processor(image, return_tensors="pt")["pixel_values"]


def infer(image_path: Path, prompt: str) -> str:
    global model
    if not model:
        load_model("checkpoints/last.ckpt")
    image = load_image(image_path).to(model.device)
    return model.generate(
        image,
        prompt,
        max_new_tokens=64,
        do_sample=True,
        num_beams=4,
    )


with gr.Blocks() as demo:
    image = gr.Image(label="Image", type="filepath")
    prompt = gr.Textbox(label="Prompt", placeholder="What is in the image?")
    button = gr.Button("Get answer!")
    output = gr.Textbox(label="Output")
    button.click(fn=infer, inputs=[image, prompt], outputs=output, api_name="image-qa")

demo.launch(server_name="0.0.0.0")
