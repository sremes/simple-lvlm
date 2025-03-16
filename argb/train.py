"""Trainer for VQVAE model."""

from argb import ArgbPixelEncoder

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode
from torch.utils.tensorboard import SummaryWriter

import argparse
from functools import partial
import os


def create_dataloader(dataset_path: str, batch_size: int, dtype: torch.dtype):
    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(dtype, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(
        dataset_path,
        transform=transform,
        loader=partial(
            decode_image, mode=ImageReadMode.RGB, apply_exif_orientation=True
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    return dataloader


def train(args):
    dataloader = create_dataloader(args.dataset_path, args.batch_size, args.dtype)
    model = ArgbPixelEncoder().to(device="cuda", dtype=args.dtype)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, fused=True)
    writer = SummaryWriter()
    model = torch.compile(model)
    for epoch in range(1):
        for iter, batch in enumerate(dataloader):
            optimizer.zero_grad()
            img, _ = batch
            img = img.to(device="cuda", dtype=args.dtype)
            with torch.autocast(device_type="cuda"):
                img_recon, loss = model(img)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch} iter: {iter} loss: {loss.item()}")
            writer.add_scalar("loss", loss.item(), epoch * len(dataloader) + iter)
            if iter % 50 == 0:
                grid = torchvision.utils.make_grid([img[0], img_recon[0]], nrow=1)
                writer.add_image("img_recon", grid, epoch * len(dataloader) + iter)
            if iter % 500 == 0:
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save(
                    getattr(model, "_orig_mod", model).state_dict(),
                    f"checkpoints/argb_epoch{epoch:02d}-step{iter:06d}-loss{loss.item():.3f}.pth",
                )
        # save after full epoch
        torch.save(
            model.state_dict(),
            f"checkpoints/argb_epoch{epoch}-loss{loss.item():.3f}.pth",
        )
    writer.close()


def str2dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str2dtype, default="float32")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_path", type=str, default="/stash/ecoset/train")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
