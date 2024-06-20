import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from flash_attn import flash_attn_func
import lpips

from functools import partial


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv(x),
            "b n (qkv h c) -> qkv b n h c",
            qkv=3,
            h=self.num_heads,
            c=self.head_dim,
        ).unbind(0)
        # x = flash_attn_func(q, k, v, causal=False, softmax_scale=self.scale)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        ).transpose(1, 2)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-4)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        embed_dim: int = 768,
        num_heads: int = 8,
        patch_size: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(
            in_chans=in_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(2)
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        img_size: int = 224,
        embed_dim: int = 768,
        num_heads: int = 8,
        patch_size: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            *[
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(2)
            ]
        )
        self.upsample = nn.ConvTranspose2d(
            embed_dim,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=self.img_size // self.patch_size,
            w=self.img_size // self.patch_size,
        )
        x = self.upsample(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 8192,
        in_channels: int = 3,
        img_size: int = 224,
        embed_dim: int = 768,
        num_heads: int = 12,  # head dim = embed_dim (768) // num_heads  (12)= 64
        patch_size: int = 8,
        beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, img_size, embed_dim, num_heads, patch_size)
        self.decoder = Decoder(in_channels, img_size, embed_dim, num_heads, patch_size)
        self.perceptual_loss = lpips.LPIPS(net="alex", verbose=False)
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.embeddings.weight.data.normal_(std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Follow notation of https://arxiv.org/pdf/1711.00937"""
        # encode and decode
        z_e = self.encoder(image)
        z_q = self.find_closest_embedding(z_e)
        # calculate vq loss
        loss_vq = ((z_e.detach() - z_q) ** 2).mean() + self.beta * (
            (z_e - z_q.detach()) ** 2
        ).mean()
        # preserve gradients
        # z_q = z_e + (z_q - z_e).detach()
        # decode from the latent
        x = self.decoder(z_q)

        # recon losses
        loss_mse = ((image - x) ** 2).mean()
        loss_abs = (image - x).abs().mean()
        loss_perceptual = self.perceptual_loss(
            x.clamp(-1, 1), image.clamp(-1, 1)
        ).mean()

        # total loss
        loss = loss_vq + loss_mse + loss_abs + loss_perceptual

        return x, loss

    def find_closest_embedding(self, z_e: torch.Tensor) -> torch.Tensor:
        """Find the closest embeddings to z_e."""
        z_e_shape = z_e.shape
        z_e = rearrange(z_e, "b n c -> (b n) c")
        distances = torch.cdist(z_e, self.embeddings.weight)
        indices = torch.argmin(distances, dim=1)
        embeddings = self.embeddings(indices).view(z_e_shape)
        return embeddings


def init_weights(
    module: nn.Linear | nn.Conv2d | nn.LayerNorm | nn.Embedding,
    initializer_range: float = 0.02,
) -> None:
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
        # `trunc_normal_cpu` not implemented in `half` issues
        module.weight.data = nn.init.trunc_normal_(
            module.weight.data.to(torch.float32),
            mean=0.0,
            std=initializer_range,
        ).to(module.weight.dtype)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(
            module.weight.data.to(torch.float32),
            mean=0.0,
            std=initializer_range,
        ).to(module.weight.dtype)


if __name__ == "__main__":
    dtype = torch.bfloat16
    grad_scale = 1
    model = VQVAE(num_embeddings=10).to(device="cuda", dtype=dtype)
    model.apply(partial(init_weights, initializer_range=0.02))
    image = torch.ones(1, 3, 224, 224, device="cuda", dtype=dtype)
    print("image.shape:", image.shape)
    output, loss = model(image)
    print("output.shape:", output.shape)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for iter in range(1000):
        optim.zero_grad()
        _, loss = model(image)
        (grad_scale * loss).backward()
        for name, param in model.named_parameters():
            param.grad.data = param.grad.data / grad_scale
            param.grad.data = torch.nan_to_num(param.grad.data, 0.0)
            torch.nn.utils.clip_grad_norm_(
                param, 1.0, norm_type=2, error_if_nonfinite=True
            )
            # print("grad norm:", name, param.grad.data.norm().item())
        optim.step()
        print(f"iter: {iter}, loss: {loss.item()}")
    print("Done!")
