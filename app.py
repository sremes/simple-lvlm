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


def infer(
    image_path: Path,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.75,
    top_p: float = 0.95,
    num_beams: int = 4,
    length_penalty: float = -1.0,
    repetition_penalty: float = 1.2,
) -> str:
    global model
    if not model:
        load_model("checkpoints/last.ckpt")
    image = load_image(image_path).to(model.device)
    return model.generate(
        image,
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
    )


with gr.Blocks() as demo:
    image = gr.Image(label="Image", type="filepath")
    prompt = gr.Textbox(label="Question", placeholder="What is in the image?")
    max_tokens = gr.Slider(1, 256, 128, step=1, label="Max tokens")
    temperature = gr.Slider(0.0, 1.0, 0.75, step=0.05, label="Temperature")
    top_p = gr.Slider(0.0, 1.0, 0.95, step=0.05, label="Top-P")
    num_beams = gr.Slider(1, 8, 4, step=1, label="Number of beams")
    length_penalty = gr.Slider(-5.0, 5.0, -1.0, step=0.1, label="Length penalty")
    repetition_penalty = gr.Slider(-5.0, 5.0, 1.2, step=0.1, label="Repetition penalty")
    button = gr.Button("Get answer!")
    output = gr.Textbox(label="Output")
    button.click(
        fn=infer,
        inputs=[
            image,
            prompt,
            max_tokens,
            temperature,
            top_p,
            num_beams,
            length_penalty,
            repetition_penalty,
        ],
        outputs=output,
        api_name="image-qa",
    )

demo.launch(server_name="0.0.0.0")
