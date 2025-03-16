import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from flash_attn import flash_attn_func

from argb.argb import ArgbPixelEncoder

from functools import partial
from typing import Optional

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()
torch.set_float32_matmul_precision("medium")


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
        ## FLASHATTN
        q, k, v = rearrange(
            self.qkv(x),
            "b n (qkv h c) -> qkv b n h c",
            qkv=3,
            h=self.num_heads,
            c=self.head_dim,
        ).unbind(0)
        x = flash_attn_func(q, k, v, causal=False, softmax_scale=self.scale)
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
        num_heads: int = 24,
        patch_size: int = 8,
        num_layers: int = 8,
        quant_dim: int = 32,
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
        self.position_embeddings = nn.Parameter(
            0.02 * torch.randn(1, self.patch_embed.num_patches, embed_dim)
        )
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, quant_dim), nn.LayerNorm(quant_dim, eps=1e-4)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)
        x = x + self.position_embeddings
        x = self.encoder(x)
        return self.proj(x)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        img_size: int = 224,
        embed_dim: int = 768,
        num_heads: int = 24,
        patch_size: int = 8,
        num_layers: int = 8,
        quant_dim: int = 32,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Linear(quant_dim, embed_dim)
        self.decoder = nn.Sequential(
            *[
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        num_patches = (img_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            0.02 * torch.randn(1, num_patches, embed_dim)
        )
        self.to_patch_pixels = nn.Linear(
            embed_dim, patch_size * patch_size * out_channels
        )
        # self.refine = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        # )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.position_embeddings
        x = self.decoder(x)
        x = self.dropout(x)
        x = rearrange(
            self.to_patch_pixels(x),
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.img_size // self.patch_size,
            w=self.img_size // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )
        # x = self.refine(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 8192,
        in_channels: int = 3,
        img_size: int = 224,
        embed_dim: int = 768,
        quant_dim: int = 64,
        num_heads: int = 12,  # head dim = embed_dim (768) // num_heads (12) = 64
        patch_size: int = 8,
        alpha: float = 10.0,
        beta: float = 0.1,
        sync_nu: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, img_size, embed_dim, num_heads, patch_size, quant_dim=quant_dim
        )
        self.decoder = Decoder(
            in_channels, img_size, embed_dim, num_heads, patch_size, quant_dim=quant_dim
        )
        # load pretrained model for reconstruct loss with frozen weights
        # self.recon_loss = ArgbPixelEncoder.from_pretrained("argb/argb-ecoset-224px.pth")
        # self.recon_loss.eval()
        # for param in self.recon_loss.parameters():
        # param.requires_grad = False
        # assert self.recon_loss.training == False
        self.alpha = alpha
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, quant_dim)
        self.embedding_mean = nn.Parameter(
            torch.zeros_like(self.embeddings.weight.mean(0, keepdim=True))
        )
        self.embedding_std = nn.Parameter(
            torch.log(torch.ones_like(self.embeddings.weight.mean(0, keepdim=True)))
        )
        self.sync_nu = sync_nu
        if num_classes is not None:
            self.classifier = nn.Linear(quant_dim, num_classes)

    def forward(
        self, image: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Follow notation of https://arxiv.org/pdf/1711.00937"""
        # encode
        z_e = self.encoder(image)
        # quantize
        z_q = self.find_closest_embedding(z_e)
        # calculate vq loss
        loss_vq = ((1 - self.beta) * (z_e.detach() - z_q) ** 2).mean() + self.beta * (
            (z_e - z_q.detach()) ** 2
        ).mean()
        # preserve gradients (straight-through gradient estimator) with the synchronized update rule
        z_q = z_e + (z_q - z_e).detach() + self.sync_nu * (z_q - z_q.detach())
        # decode from the latent
        z_q = F.dropout(z_q, p=0.1, training=self.training)
        x = self.decoder(z_q)

        # calculate reconstruction loss
        loss_recon = (
            # 10 * self.recon_loss.loss(x, image)
            F.mse_loss(x, image)
            + F.l1_loss(x, image)
        )

        # optionally compute a classification loss
        if label is not None:
            y = self.classifier(z_q.mean(1))
            loss_class = F.cross_entropy(y, label)
            return x, self.alpha * loss_vq + loss_recon + loss_class

        # total loss without classification loss
        loss = self.alpha * loss_vq + loss_recon
        return x, loss

    def find_closest_embedding(self, z_e: torch.Tensor) -> torch.Tensor:
        """Find the closest embeddings to z_e."""
        z_e_shape = z_e.shape
        z_e = rearrange(z_e, "b n c -> (b n) c")
        # affine parameterization of the codebook embeddings
        embedding_std = self.embedding_std.exp()
        with torch.no_grad():
            all_embeddings = (
                embedding_std * self.embeddings.weight + self.embedding_mean
            )
            all_embeddings = F.normalize(all_embeddings, dim=1)
            distances = torch.cdist(F.normalize(z_e, dim=1), all_embeddings)
            if self.training:
                distances = distances * (0.95 + 0.1 * torch.rand_like(distances))
            indices = torch.argmin(distances, dim=1)
        # return the closest embeddings transformed by the affine parameters
        embeddings = (
            embedding_std * self.embeddings(indices) + self.embedding_mean
        ).view(z_e_shape)
        return embeddings


def init_weights(
    module: nn.Linear | nn.Conv2d | nn.LayerNorm | nn.Embedding,
    initializer_range: float = 0.02,
) -> None:
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.reset_parameters()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Embedding):
        num_embeddings, _ = module.weight.size()
        nn.init.uniform_(module.weight, -1 / num_embeddings, 1 / num_embeddings)
        with torch.no_grad():
            module.weight /= module.weight.norm(dim=1, keepdim=True)


if __name__ == "__main__":
    dtype = torch.float16
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
