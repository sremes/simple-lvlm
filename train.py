from model import LVLM

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from transformers.image_processing_utils import BaseImageProcessor
import bitsandbytes as bnb
import PIL.Image
import tqdm

import json
import logging
from pathlib import Path
import random
import shutil
from itertools import pairwise
from functools import partial

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename="train.log",
    encoding="utf-8",
    level=logging.INFO,
)
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


class Ecoset(torchvision.datasets.ImageFolder):
    prompts: tuple[str] = (
        "What does the image depict?",
        "What can be seen in the image?",
        "What do you see in this image?",
        "Please describe the content of this image.",
        "Identify and describe the main object depicted in this image.",
        "What is the main object visible in the image?",
    )
    output_templates: tuple[str] = (
        "This image depicts {0} {1}.",
        "There is {0} {1} in the image.",
        "In this image, there is {0} {1}.",
        "I can see there is {0} {1} in the image.",
        "The main object in this image is {0} {1}.",
        "{1}.",
        "Object: {1}."
    )
    vowels: tuple[str] = ("a", "e", "i", "o", "u")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image, target = super().__getitem__(index)
        target_name = self.classes[target].split("_")[-1]
        article = "an" if target_name[0] in self.vowels else "a"
        return {
            "image": image["pixel_values"].to(torch.bfloat16).squeeze(),
            "text_input": random.sample(self.prompts, 1)[0],
            "text_output": random.sample(self.output_templates, 1)[0].format(
                article, target_name
            ),
        }


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: Path
) -> None:
    param_has_grad = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if not param_has_grad.get(k, False):
            del state_dict[k]
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    *path_parts, last = path.parts
    step_name = last.replace("last", f"step_{step}")
    step_path = Path(*path_parts, step_name)
    shutil.copy(path, step_path)


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path
) -> None:
    logging.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except ValueError as e:
        logging.info(f"Not loading optimizer state: {e}")


def setup_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = bnb.optim.AdamW8bit(trainable_parameters, lr=lr, eps=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 5000, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose=False
    )
    return optimizer, lr_scheduler


@torch.inference_mode()
def compute_validation_loss(
    model: torch.nn.Module, val_loader: DataLoader
) -> torch.Tensor:
    """Compute the loss over validation data batches in inference mode."""
    val_loss = torch.tensor(0.0, dtype=torch.bfloat16, device=model.device)
    for sample in val_loader:
        loss = model(
            image=sample["image"].to(model.device),
            text_input=sample["text_input"],
            text_output=sample["text_output"],
        )
        val_loss += loss / len(val_loader)
    return val_loss


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 1,
    grad_accumulation_steps: int = 4,
    save_every_n_step: int = 500,
) -> None:
    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    logging.info(
        f"Number of trainable parameters: {sum(p.numel() for p in trainable_parameters)}"
    )
    # optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-5, fused=True, eps=1e-4)
    ckpt_path = Path("checkpoints/last.ckpt")
    if ckpt_path.exists():
        load_checkpoint(model, optimizer, ckpt_path)
    logging.info("starting training")
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        logging.info(f"Epoch: {epoch}")
        for step, data in enumerate(bar := tqdm.tqdm(data_loader)):
            loss = model(
                image=data["image"].to(model.device),
                text_input=data["text_input"],
                text_output=data["text_output"],
            )
            # (loss / grad_accumulation_steps).backward()
            accelerator.backward(loss / grad_accumulation_steps)
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step(step)
                optimizer.zero_grad()

            bar.set_description(f"Training progress")
            bar.set_postfix(
                {
                    "iter": step,
                    "loss": loss.cpu().item(),
                    "lr": lr_scheduler.get_last_lr(),
                }
            )
            writer.add_scalar("loss", loss, step)
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], step)
            if step % save_every_n_step == 0:
                logging.info(f"saving checkpoint at epoch {epoch}, iter {step}")
                save_checkpoint(model, optimizer, step, ckpt_path)

                val_loss = compute_validation_loss(model, val_loader)
                writer.add_scalar("val_loss", val_loss.cpu().item(), step)
                writer.flush()


if __name__ == "__main__":
    from accelerate import Accelerator

    accelerator = Accelerator()

    model = LVLM(
        language_model="stabilityai/stablelm-zephyr-3b",
        vision_model="openai/clip-vit-large-patch14",
        device=accelerator.device,
    )
    # training data
    llava_dataset = LLaVADataset(
        annotations_file="data/llava_v1_5_mix665k_train.json",
        image_dir=Path("/stash/datasets"),
        image_processor=model.image_processor,
    )
    ecoset_dataset = Ecoset(
       root="/stash/ecoset/train",
       transform=partial(model.image_processor, return_tensors="pt"),
    )
    dataset = ConcatDataset((llava_dataset, ecoset_dataset))
    data_loader = DataLoader(
        dataset, batch_size=6, shuffle=True, num_workers=2, pin_memory=True
    )
    # validation data
    val_dataset = LLaVADataset(
        annotations_file="data/llava_v1_5_mix665k_val.json",
        image_dir=Path("/stash/datasets"),
        image_processor=model.image_processor,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True
    )
    # setup training
    optimizer, scheduler = setup_optimizer(model, lr=1e-5)
    model, optimizer, data_loader, scheduler, val_loader = accelerator.prepare(
        model, optimizer, data_loader, scheduler, val_loader
    )

    train(
        model,
        optimizer,
        scheduler,
        data_loader,
        val_loader,
        num_epochs=1,
        grad_accumulation_steps=8,
    )
