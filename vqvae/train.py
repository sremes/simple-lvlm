"""Trainer for VQVAE model."""

from vqvae import VQVAE, init_weights

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode
from torch.utils.tensorboard import SummaryWriter

import argparse
from functools import partial
import os
from itertools import islice

# torch._inductor.config.max_autotune = True
# torch._inductor.max_autotune_gemm_backends = "TRITON"
# torch._inductor.max_autotune_conv_backends = "TRITON"
# torch._inductor.config.coordinate_descent_tuning = True
# # torch._inductor.config.freezing = True
# torch._inductor.config.layout_optimization = True
# torch._inductor.config.cpp_wrapper = True
# torch._inductor.config.rocm.compile_opt_level = "-Ofast"
# os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"


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


def log_model_summary(model: torch.nn.Module, writer: SummaryWriter):
    print(model)
    writer.add_text("model_summary", str(model))
    writer.add_text("model_params", str(sum(p.numel() for p in model.parameters())))
    writer.add_text(
        "model_trainable_params",
        str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )


def train(args):
    # create dataloader and model
    dataloader = create_dataloader(args.dataset_path, args.batch_size, args.dtype)
    # num_classes = len(dataloader.dataset.classes)
    # model = VQVAE(num_classes=num_classes).to(device="cuda", dtype=args.dtype)
    model = VQVAE().to(device="cuda", dtype=args.dtype)
    # create optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    # load state or init
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as e:
            print("warning: ", e)
            pass
        global_step = checkpoint["step"] if "step" in checkpoint else 0
        del checkpoint
    else:
        model.apply(init_weights)
        global_step = 0
    # setup tensorboard
    writer = SummaryWriter()
    writer.add_hparams(
        {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dtype": str(args.dtype),
            "dataset_path": args.dataset_path,
        },
        {},
    )
    log_model_summary(model, writer)
    # start training
    last_epoch = global_step // len(dataloader)
    last_step = global_step % len(dataloader)
    for epoch in range(last_epoch, 1):
        # first skip last_step steps from dataloader
        if epoch == last_epoch and last_step > 0:
            data_iterator = enumerate(
                islice(dataloader, last_step, None), start=last_step
            )
        else:
            data_iterator = enumerate(dataloader)
        for iter, batch in data_iterator:
            global_step = epoch * len(dataloader) + iter
            optimizer.zero_grad()
            # img, labels = batch
            img, _ = batch
            img = img.to(device="cuda", dtype=args.dtype)
            # labels = labels.to(device="cuda", dtype=torch.long)
            with torch.autocast(
                device_type="cuda", enabled=(args.dtype == torch.float32)
            ):
                # img_recon, loss = model(img, labels)
                img_recon, loss = model(img)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"epoch: {epoch} iter: {iter} loss: {loss.item()}")
            writer.add_scalar("loss", loss.item(), global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            if iter % 50 == 0:
                grid = torchvision.utils.make_grid(
                    [img[0], img_recon[0]], nrow=1, normalize=True, scale_each=True
                )
                writer.add_image("img_recon", grid, global_step)
            if iter % 500 == 0:
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                    },
                    f"checkpoints/vqvae_epoch{epoch:02d}-step{iter:06d}-loss{loss.item():.3f}.pth",
                )
        # save after full epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            },
            f"checkpoints/vqvae_epoch{epoch}-loss{loss.item():.3f}.pth",
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
    parser.add_argument("--dtype", type=str2dtype, default="bfloat16")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--dataset_path", type=str, default="/stash/ecoset/train")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
