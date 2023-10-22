from model import LVLM

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.image_processing_utils import BaseImageProcessor
import PIL.Image
import tqdm

import json
import logging
from pathlib import Path
import random
from itertools import pairwise

logging.basicConfig(filename="train.log", encoding="utf-8", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())


class AnnotationException(Exception):
    pass


class LLaVADataset(Dataset):
    def __init__(
        self,
        annotations_file: Path,
        image_dir: Path,
        image_processor: BaseImageProcessor,
    ) -> None:
        # load annotations
        with open(annotations_file, "rt") as f:
            self.annotations: list[dict] = json.load(f)
        self.image_dir = image_dir
        self.image_processor = image_processor

    @staticmethod
    def sample_qa_pair(conversations: list[dict[str, str]]) -> tuple[str, str]:
        qa_pairs = list(pairwise(conversations))[::2]
        qa_pair = random.sample(qa_pairs, 1)[0]
        question = qa_pair[0]["value"].replace("<image>", "").replace("\n", "")
        answer = qa_pair[1]["value"].replace("\n", " ")
        return question, answer

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        while True:
            try:
                annotation = self.annotations[index]
                if "image" not in annotation.keys():
                    raise AnnotationException(f"annotation has no image: {annotation}")
                image_path = Path(self.image_dir, annotation["image"])
                if not image_path.exists():
                    raise AnnotationException(f"image not found: {image_path}")
            except AnnotationException as e:
                logging.debug(e)
                index = random.sample(range(len(self)), 1)[0]
                continue
            else:
                break
        image = PIL.Image.open(image_path)
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        question, answer = self.sample_qa_pair(annotation["conversations"])
        return {
            "image": image.squeeze().to(torch.bfloat16),
            "text_input": question,
            "text_output": answer,
        }

    def __len__(self) -> int:
        return len(self.annotations)


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path
) -> None:
    param_has_grad = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if not param_has_grad.get(k, False):
            del state_dict[k]
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path
) -> None:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def train(model: torch.nn.Module, data_loader: DataLoader, num_epochs: int = 1) -> None:
    optimizer = torch.optim.AdamW(
        filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4
    )
    ckpt_path = Path("checkpoints/last.ckpt")
    if ckpt_path.exists():
        load_checkpoint(model, optimizer, ckpt_path)
    for epoch in range(num_epochs):
        logging.info(f"Epoch: {epoch}")
        for step, data in enumerate(bar := tqdm.tqdm(data_loader)):
            optimizer.zero_grad()
            loss = model(
                image=data["image"].to(model.device),
                text_input=data["text_input"],
                text_output=data["text_output"],
            )
            loss.backward()
            optimizer.step()

            bar.set_description(f"Training progress")
            bar.set_postfix({"iter": step, "loss": loss.cpu().item()})
            if step % 1000 == 0:
                logging.info(f"saving checkpoint at epoch {epoch}, iter {step}")
                save_checkpoint(model, optimizer, epoch, ckpt_path)


if __name__ == "__main__":
    model = LVLM(
        language_model="stabilityai/stablelm-3b-4e1t",
        vision_model="openai/clip-vit-large-patch14",
    )
    dataset = LLaVADataset(
        annotations_file="data/llava_v1_5_mix665k.json",
        image_dir=Path("/stash/datasets"),
        image_processor=model.image_processor,
    )
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    train(model, data_loader, num_epochs=1)
